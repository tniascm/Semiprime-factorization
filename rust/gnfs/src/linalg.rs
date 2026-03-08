use crate::arith::QuadCharSet;
use crate::types::BitRow;
use crate::linalg_wiedemann::{SparseMatrix, wiedemann_nullspace_multi};

/// Build the GF(2) exponent matrix from relations.
///
/// Column layout places sparse columns FIRST so GE eliminates them early,
/// producing longer dependencies from the dense FB columns:
///
/// [SQ | rat_LP | alg_LP | sign_rat | rat_FB | sign_alg | alg_pairs | alg_HD | QC]
///
/// - SQ columns: one per unique special-q (prime, root) pair
/// - rat_LP columns: one per unique rational large prime
/// - alg_LP columns: one per unique algebraic large prime (prime, root)
/// - sign_rat: rational sign bit
/// - rat_FB: rational factor base exponents mod 2
/// - sign_alg: algebraic sign bit
/// - alg_pairs: per-(prime,root) algebraic pair exponents mod 2
/// - alg_HD: higher-degree ideal exponents mod 2
/// - QC: quadratic character Legendre symbols
pub fn build_matrix(
    relations: &[crate::types::Relation],
    rat_fb_size: usize,
    alg_pair_count: usize,
    alg_hd_count: usize,
    quad_chars: &QuadCharSet,
) -> (Vec<BitRow>, usize) {
    // Collect unique special-q (prime, root) pairs and assign column indices
    let mut sq_pairs: Vec<(u64, u64)> = relations.iter().filter_map(|r| r.special_q).collect();
    sq_pairs.sort_unstable();
    sq_pairs.dedup();

    // Collect unique rational large primes
    let mut rat_lps: Vec<u64> = relations.iter().filter_map(|r| r.rat_lp).collect();
    rat_lps.sort_unstable();
    rat_lps.dedup();

    // Collect unique algebraic large prime (prime, root) pairs
    let mut alg_lps: Vec<(u64, u64)> = relations.iter().filter_map(|r| r.alg_lp).collect();
    alg_lps.sort_unstable();
    alg_lps.dedup();

    let n_sq = sq_pairs.len();
    let n_rat_lp = rat_lps.len();
    let n_alg_lp = alg_lps.len();
    let n_qc = quad_chars.primes.len();
    // Column layout: sparse columns FIRST so GE eliminates them early.
    // This produces longer dependencies from the dense FB columns.
    // [SQ | rat_LP | alg_LP | sign_rat | rat_FB | sign_alg | alg_pairs | alg_HD | QC]
    let sq_col_base = 0;
    let rat_lp_col_base = n_sq;
    let alg_lp_col_base = rat_lp_col_base + n_rat_lp;
    let dense_col_base = alg_lp_col_base + n_alg_lp;
    let qc_col_base = dense_col_base + 2 + rat_fb_size + alg_pair_count + alg_hd_count;
    let ncols = qc_col_base + n_qc;

    let mut rows = Vec::with_capacity(relations.len());

    for rel in relations {
        let mut row = BitRow::new(ncols);

        // Dense columns start at dense_col_base
        if rel.rational_sign_negative {
            row.set(dense_col_base);
        }

        for &(idx, exp) in &rel.rational_factors {
            if exp % 2 == 1 {
                row.set(dense_col_base + 1 + idx as usize);
            }
        }

        if rel.algebraic_sign_negative {
            row.set(dense_col_base + 1 + rat_fb_size);
        }

        // algebraic_factors stores (flat_index, exp) where flat_index is:
        // - [0, alg_pair_count): per-(prime,root) pair columns
        // - [alg_pair_count, alg_pair_count+alg_hd_count): HD ideal columns
        for &(flat_idx, exp) in &rel.algebraic_factors {
            if exp % 2 == 1 {
                row.set(dense_col_base + 2 + rat_fb_size + flat_idx as usize);
            }
        }

        // Sparse columns at the front of the matrix
        // Special-q column: each relation from special-q (q, r) has exponent 1 (odd)
        if let Some(sq) = rel.special_q {
            if let Ok(idx) = sq_pairs.binary_search(&sq) {
                row.set(sq_col_base + idx);
            }
        }

        // Rational large prime: exponent 1 (odd)
        if let Some(lp) = rel.rat_lp {
            if let Ok(idx) = rat_lps.binary_search(&lp) {
                row.set(rat_lp_col_base + idx);
            }
        }

        // Algebraic large prime: exponent 1 (odd)
        if let Some(lp) = rel.alg_lp {
            if let Ok(idx) = alg_lps.binary_search(&lp) {
                row.set(alg_lp_col_base + idx);
            }
        }

        // Quadratic characters: Legendre symbol ((a - b*r) / q)
        for (i, (&q, &r)) in quad_chars
            .primes
            .iter()
            .zip(quad_chars.roots.iter())
            .enumerate()
        {
            let q_i = q as i128;
            let val = (rel.a as i128 - rel.b as i128 * r as i128).rem_euclid(q_i) as u64;
            if val == 0 {
                continue; // q divides (a - b*r), treat as +1
            }
            let ls = crate::arith::legendre_symbol(val, q);
            if ls == q - 1 {
                // Legendre symbol is -1 → set bit (odd in GF(2))
                row.set(qc_col_base + i);
            }
        }

        rows.push(row);
    }

    (rows, ncols)
}

/// Dense GF(2) Gaussian elimination to find null-space vectors.
///
/// Returns a list of dependencies, where each dependency is a list of row indices
/// whose XOR is the zero vector (i.e., the corresponding relations form a square product).
///
/// Uses BitRow-based history tracking (O(nrows/64) per XOR) instead of Vec<usize>
/// (O(n) per symmetric difference), which is critical for performance on large matrices.
pub fn find_dependencies(rows: &[BitRow], ncols: usize) -> Vec<Vec<usize>> {
    let nrows = rows.len();
    if nrows == 0 || ncols == 0 {
        return vec![];
    }

    // Heuristically switch to Wiedemann for large matrices to avoid O(N^3) GE.
    if nrows > 5000 && ncols > 5000 {
        let sparse = SparseMatrix::from_bitrows(rows, ncols);
        let deps = wiedemann_nullspace_multi(&sparse);
        if !deps.is_empty() {
            return deps;
        }
    }

    let mut matrix: Vec<BitRow> = rows.to_vec();
    // Track which original rows compose each current row using BitRows
    let mut history: Vec<BitRow> = (0..nrows)
        .map(|i| {
            let mut h = BitRow::new(nrows);
            h.set(i);
            h
        })
        .collect();

    let mut pivot_row = 0;
    for col in 0..ncols {
        let mut found = None;
        for r in pivot_row..nrows {
            if matrix[r].get(col) {
                found = Some(r);
                break;
            }
        }

        let pr = match found {
            Some(r) => r,
            None => continue,
        };

        if pr != pivot_row {
            matrix.swap(pr, pivot_row);
            history.swap(pr, pivot_row);
        }

        // Forward-only elimination: only reduce rows BELOW the pivot.
        // For null-space finding, backward elimination is unnecessary — pivot rows
        // always retain their pivot bit and never become zero, so the same
        // dependencies appear as zero rows below the final pivot position.
        let pivot_data = matrix[pivot_row].clone();
        let pivot_hist = history[pivot_row].clone();
        for r in (pivot_row + 1)..nrows {
            if matrix[r].get(col) {
                matrix[r].xor_with(&pivot_data);
                history[r].xor_with(&pivot_hist);
            }
        }

        pivot_row += 1;
    }

    // Extract dependencies: zero rows whose history has ≥2 original rows
    let mut deps = Vec::new();
    for r in 0..nrows {
        if matrix[r].is_zero() {
            let dep: Vec<usize> = (0..nrows).filter(|&i| history[r].get(i)).collect();
            if dep.len() >= 2 {
                deps.push(dep);
            }
        }
    }

    deps
}

/// Combine null-space basis vectors into randomized dependencies.
///
/// GE produces short, correlated null-space vectors that often give γ(m) ≡ ±x (mod N)
/// in the sqrt stage. Random linear combinations (XOR of random subsets) produce
/// longer, decorrelated dependencies with ~50% probability of nontrivial factors.
///
/// Returns the original basis deps plus `n_random` random combinations, each
/// combining `k` randomly chosen basis vectors.
pub fn randomize_dependencies(
    basis_deps: &[Vec<usize>],
    n_random: usize,
    k: usize,
    seed: u64,
) -> Vec<Vec<usize>> {
    let n = basis_deps.len();
    // Need at least 3 basis vectors to create nontrivial randomized XOR combos.
    if n < 3 {
        return basis_deps.to_vec();
    }

    // Find the universe size (max relation index + 1) for BitRow representation
    let universe = basis_deps
        .iter()
        .flat_map(|d| d.iter())
        .copied()
        .max()
        .unwrap_or(0)
        + 1;

    // Convert basis deps to BitRow representation for O(nwords) XOR
    let basis_bits: Vec<BitRow> = basis_deps
        .iter()
        .map(|dep| {
            let mut br = BitRow::new(universe);
            for &idx in dep {
                br.set(idx);
            }
            br
        })
        .collect();

    let mut all_deps = basis_deps.to_vec();
    // Track seen deps as BitRows for fast dedup
    let mut seen_bits: Vec<BitRow> = basis_bits.clone();
    let mut rng_state = seed;
    // Important: never choose all basis vectors every time, otherwise for small
    // nullspaces (n <= k) randomization collapses to one repeated dependency.
    let k_eff = k.clamp(2, n - 1);

    // Reusable buffer for XOR combination
    let mut combined = BitRow::new(universe);

    for _ in 0..n_random {
        // Simple LCG for deterministic PRNG.
        // Randomize subset size in [2, k_eff] to increase diversity.
        rng_state = rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let target_k = if k_eff > 2 {
            2 + ((rng_state >> 17) as usize % (k_eff - 1))
        } else {
            2
        };

        // Choose target_k distinct basis indices
        let mut chosen_count = 0usize;
        let mut chosen_mask = BitRow::new(n);
        while chosen_count < target_k {
            rng_state = rng_state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let idx = (rng_state >> 33) as usize % n;
            if !chosen_mask.get(idx) {
                chosen_mask.set(idx);
                chosen_count += 1;
            }
        }

        // XOR of the chosen basis vectors using BitRow — O(nwords) per XOR
        for w in combined.bits.iter_mut() {
            *w = 0;
        }
        for (basis_idx, basis_br) in basis_bits.iter().enumerate() {
            if chosen_mask.get(basis_idx) {
                combined.xor_with(basis_br);
            }
        }

        // Count set bits to check >= 3
        let popcount: u32 = combined.bits.iter().map(|w| w.count_ones()).sum();
        if popcount >= 3 {
            // Check if this combination is new (not seen before)
            let is_new = !seen_bits.iter().any(|s| s.bits == combined.bits);
            if is_new {
                // Extract indices from BitRow
                let mut dep = Vec::with_capacity(popcount as usize);
                for (wi, &word) in combined.bits.iter().enumerate() {
                    let mut w = word;
                    while w != 0 {
                        let bit = w.trailing_zeros();
                        dep.push(wi * 64 + bit as usize);
                        w &= w - 1; // clear lowest set bit
                    }
                }
                seen_bits.push(combined.clone());
                all_deps.push(dep);
            }
        }
    }

    all_deps
}

/// Iterative weight-1/weight-2 column pre-elimination pass followed by dense GE.
///
/// Before running the expensive dense Gaussian elimination, this function
/// iteratively removes lightweight columns:
///
/// 1. **Singleton removal**: columns with weight 1 cannot participate in any
///    dependency (a single row XOR is never zero). The row containing the
///    singleton is eliminated along with the column.
///
/// 2. **Weight-2 merge**: columns with weight exactly 2 in rows r1 and r2 can
///    be eliminated by XOR-ing r2 into r1 (merging all of r2's information
///    into r1) and removing r2. The column disappears because 1 XOR 1 = 0
///    in r1 and r2 is gone.
///
/// Steps 1-2 iterate until no more changes occur (each elimination can create
/// new singletons or weight-2 columns). The surviving rows/columns form a
/// compacted matrix that is passed to `find_dependencies`. Results are then
/// mapped back to original row indices.
///
/// This can dramatically reduce matrix size for NFS matrices where large-prime
/// columns are naturally sparse.
pub fn find_dependencies_with_preelim(rows: &[BitRow], ncols: usize) -> Vec<Vec<usize>> {
    let nrows = rows.len();
    if nrows == 0 || ncols == 0 {
        return vec![];
    }

    // Work on a mutable copy. Each row in `matrix` also tracks which original
    // rows it represents (via XOR merges). We store this as a Vec<Vec<usize>>
    // since during pre-elimination the merges are few and lists stay short.
    let mut matrix: Vec<BitRow> = rows.to_vec();
    // history[i] = set of original row indices that have been XOR-ed into row i.
    let mut history: Vec<Vec<usize>> = (0..nrows).map(|i| vec![i]).collect();

    let mut row_alive = vec![true; nrows];
    let mut col_alive = vec![true; ncols];

    // Pre-elimination loop
    loop {
        let mut changed = false;

        // Compute column weights over alive rows, and record which alive rows
        // contain each alive column. We only need up to 3 entries per column
        // (we care about weight 0, 1, or 2; anything >= 3 we skip).
        // col_rows[c] stores up to 3 row indices where column c is set.
        let mut col_rows: Vec<Vec<usize>> = vec![Vec::new(); ncols];
        for r in 0..nrows {
            if !row_alive[r] {
                continue;
            }
            for c in 0..ncols {
                if !col_alive[c] {
                    continue;
                }
                if matrix[r].get(c) {
                    if col_rows[c].len() < 3 {
                        col_rows[c].push(r);
                    }
                }
            }
        }

        // Pass 1: remove singleton columns (weight == 1).
        for c in 0..ncols {
            if !col_alive[c] {
                continue;
            }
            if col_rows[c].len() == 1 {
                let r = col_rows[c][0];
                if row_alive[r] {
                    row_alive[r] = false;
                    // Mark all columns that this row participates in as
                    // potentially changed (their weight decreased by 1).
                    // We don't need to update col_rows here; the outer loop
                    // will recompute on the next iteration.
                    col_alive[c] = false;
                    changed = true;
                }
            }
        }

        // Pass 2: merge weight-2 columns.
        // IMPORTANT: each merge modifies a row's bits, which can change other
        // columns' weights. We must recompute all column weights after each
        // merge, so we break after the first successful merge.
        for c in 0..ncols {
            if !col_alive[c] {
                continue;
            }
            // Compute actual weight from scratch (pass 1 may have killed rows,
            // and col_rows[] may be stale from previous merges).
            let mut alive_in_col = Vec::new();
            for r in 0..nrows {
                if row_alive[r] && matrix[r].get(c) {
                    alive_in_col.push(r);
                    if alive_in_col.len() > 2 {
                        break;
                    }
                }
            }
            if alive_in_col.len() == 2 {
                let r1 = alive_in_col[0];
                let r2 = alive_in_col[1];
                // XOR row r2 into r1, then eliminate r2.
                let r2_data = matrix[r2].clone();
                matrix[r1].xor_with(&r2_data);
                // Merge history: symmetric difference.
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
                break; // recompute all column weights
            } else if alive_in_col.is_empty() {
                col_alive[c] = false;
            } else if alive_in_col.len() == 1 {
                // Became singleton after pass 1 row removals
                row_alive[alive_in_col[0]] = false;
                col_alive[c] = false;
                changed = true;
            }
        }

        if !changed {
            break;
        }
    }

    // Build compacted matrix from surviving rows and columns.
    let surviving_rows: Vec<usize> = (0..nrows).filter(|&r| row_alive[r]).collect();
    let surviving_cols: Vec<usize> = (0..ncols).filter(|&c| col_alive[c]).collect();

    // Check for dependencies among rows that became all-zero after pre-elimination.
    // These are valid dependencies found purely by the pre-elimination pass.
    let mut deps = Vec::new();
    for &r in &surviving_rows {
        // Check if this row is now entirely zero across all surviving columns.
        let all_zero = surviving_cols.iter().all(|&c| !matrix[r].get(c));
        if all_zero {
            let mut dep = history[r].clone();
            dep.sort_unstable();
            if dep.len() >= 2 {
                deps.push(dep);
            }
        }
    }

    // Filter out zero-rows from the set we send to GE.
    let ge_rows: Vec<usize> = surviving_rows
        .iter()
        .copied()
        .filter(|&r| !surviving_cols.iter().all(|&c| !matrix[r].get(c)))
        .collect();

    if ge_rows.is_empty() || surviving_cols.is_empty() {
        return deps;
    }

    let compact_nrows = ge_rows.len();
    let compact_ncols = surviving_cols.len();

    let mut compact_matrix: Vec<BitRow> = Vec::with_capacity(compact_nrows);
    for &r in &ge_rows {
        let mut compact_row = BitRow::new(compact_ncols);
        for (new_c, &old_c) in surviving_cols.iter().enumerate() {
            if matrix[r].get(old_c) {
                compact_row.set(new_c);
            }
        }
        compact_matrix.push(compact_row);
    }

    // Run dense GE on the compacted matrix.
    let compact_deps = find_dependencies(&compact_matrix, compact_ncols);

    // Map compact row indices back to original row indices via history.
    for cdep in compact_deps {
        // Each index in cdep refers to a row in ge_rows[].
        // That row's history gives the original row indices.
        // The dependency is the XOR (symmetric difference) of all histories.
        let mut combined = std::collections::HashSet::new();
        for &ci in &cdep {
            let orig_row = ge_rows[ci];
            for &orig_idx in &history[orig_row] {
                if !combined.remove(&orig_idx) {
                    combined.insert(orig_idx);
                }
            }
        }
        if combined.len() >= 2 {
            let mut dep: Vec<usize> = combined.into_iter().collect();
            dep.sort_unstable();
            deps.push(dep);
        }
    }

    deps
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::BitRow;

    #[test]
    fn test_bitrow_operations() {
        let mut row = BitRow::new(128);
        assert!(!row.get(0));
        row.set(0);
        assert!(row.get(0));
        row.set(64);
        assert!(row.get(64));
        row.flip(0);
        assert!(!row.get(0));
    }

    #[test]
    fn test_bitrow_xor() {
        let mut a = BitRow::new(128);
        let mut b = BitRow::new(128);
        a.set(0);
        a.set(5);
        b.set(5);
        b.set(10);
        a.xor_with(&b);
        assert!(a.get(0));
        assert!(!a.get(5));
        assert!(a.get(10));
    }

    #[test]
    fn test_gaussian_elimination_simple() {
        let ncols = 3;
        let mut rows = vec![BitRow::new(ncols), BitRow::new(ncols), BitRow::new(ncols)];
        rows[0].set(0);
        rows[0].set(2);
        rows[1].set(1);
        rows[1].set(2);
        rows[2].set(0);
        rows[2].set(1);

        let deps = find_dependencies(&rows, ncols);
        assert!(!deps.is_empty(), "Should find at least one dependency");
        for dep in &deps {
            assert!(dep.len() >= 2, "Dependency should involve multiple rows");
        }
    }

    #[test]
    fn test_gaussian_elimination_overdetermined() {
        let ncols = 3;
        let mut rows: Vec<BitRow> = (0..5).map(|_| BitRow::new(ncols)).collect();
        rows[0].set(0);
        rows[1].set(1);
        rows[2].set(2);
        rows[3].set(0);
        rows[3].set(1);
        rows[4].set(1);
        rows[4].set(2);

        let deps = find_dependencies(&rows, ncols);
        assert!(deps.len() >= 2, "Should find at least 2 dependencies");
    }

    /// Helper: verify that a dependency is valid (XOR of its rows is zero).
    fn verify_dependency(rows: &[BitRow], dep: &[usize], ncols: usize) -> bool {
        let mut combined = BitRow::new(ncols);
        for &idx in dep {
            combined.xor_with(&rows[idx]);
        }
        combined.is_zero()
    }

    #[test]
    fn test_preelim_matches_plain_ge() {
        // Same matrix as test_gaussian_elimination_simple: 3 rows, 3 cols.
        // Row 0: cols {0, 2}
        // Row 1: cols {1, 2}
        // Row 2: cols {0, 1}
        // XOR of all three = zero, so {0,1,2} is a dependency.
        let ncols = 3;
        let mut rows = vec![BitRow::new(ncols), BitRow::new(ncols), BitRow::new(ncols)];
        rows[0].set(0);
        rows[0].set(2);
        rows[1].set(1);
        rows[1].set(2);
        rows[2].set(0);
        rows[2].set(1);

        let deps = find_dependencies_with_preelim(&rows, ncols);
        assert!(!deps.is_empty(), "Should find at least one dependency");
        for dep in &deps {
            assert!(dep.len() >= 2);
            assert!(
                verify_dependency(&rows, dep, ncols),
                "Dependency {:?} does not XOR to zero",
                dep
            );
        }
    }

    #[test]
    fn test_preelim_singleton_elimination() {
        // 4 rows, 4 columns.
        // Row 0: col 0 only (singleton on col 0 -- will be eliminated)
        // Row 1: cols {1, 3}
        // Row 2: cols {2, 3}
        // Row 3: cols {1, 2}
        // After eliminating row 0 (singleton col 0), the remaining 3 rows
        // on cols {1,2,3} form a dependency: rows {1,2,3} XOR to zero.
        let ncols = 4;
        let mut rows: Vec<BitRow> = (0..4).map(|_| BitRow::new(ncols)).collect();
        rows[0].set(0);
        rows[1].set(1);
        rows[1].set(3);
        rows[2].set(2);
        rows[2].set(3);
        rows[3].set(1);
        rows[3].set(2);

        let deps = find_dependencies_with_preelim(&rows, ncols);
        assert!(!deps.is_empty(), "Should find dependency among rows 1,2,3");
        for dep in &deps {
            assert!(
                !dep.contains(&0),
                "Row 0 was a singleton and should not appear in any dependency"
            );
            assert!(verify_dependency(&rows, dep, ncols));
        }
    }

    #[test]
    fn test_preelim_weight2_merge() {
        // 4 rows, 5 columns.
        // Row 0: cols {0, 2}
        // Row 1: cols {0, 3}
        // Row 2: cols {1, 4}
        // Row 3: cols {1, 2, 3, 4}
        //
        // Col 0 has weight 2 in rows {0, 1}. Merge: row0 ^= row1 -> row0 becomes {2, 3}.
        // Row 1 eliminated. Col 0 eliminated.
        // Col 1 has weight 2 in rows {2, 3}. Merge: row2 ^= row3 -> row2 becomes {2, 3, 4} XOR {4} = {2, 3}.
        // Wait, let me recalculate. row2 = {1, 4}, row3 = {1, 2, 3, 4}.
        // After XOR: row2 becomes {2, 3}. Row 3 eliminated. Col 1 eliminated.
        // Now row0 = {2, 3} (history: {0, 1}) and row2 = {2, 3} (history: {2, 3}).
        // These are identical, so XOR = zero -> dependency {0, 1, 2, 3}.
        let ncols = 5;
        let mut rows: Vec<BitRow> = (0..4).map(|_| BitRow::new(ncols)).collect();
        rows[0].set(0);
        rows[0].set(2);
        rows[1].set(0);
        rows[1].set(3);
        rows[2].set(1);
        rows[2].set(4);
        rows[3].set(1);
        rows[3].set(2);
        rows[3].set(3);
        rows[3].set(4);

        let deps = find_dependencies_with_preelim(&rows, ncols);
        assert!(!deps.is_empty(), "Should find at least one dependency");
        for dep in &deps {
            assert!(verify_dependency(&rows, dep, ncols));
        }
        // Verify that the expected 4-row dependency exists.
        let has_full_dep = deps.iter().any(|d| {
            let mut s = d.clone();
            s.sort_unstable();
            s == vec![0, 1, 2, 3]
        });
        assert!(has_full_dep, "Expected dependency {{0,1,2,3}}");
    }

    #[test]
    fn test_preelim_iterative_cascade() {
        // Demonstrate that singleton elimination cascades.
        // 3 rows, 4 columns.
        // Row 0: cols {0, 1}     -- col 0 is singleton (only row 0 has it)
        // Row 1: cols {1, 2, 3}
        // Row 2: cols {2, 3}
        //
        // Iteration 1: col 0 singleton -> eliminate row 0.
        //   Now col 1 has weight 1 (only row 1). -> singleton cascade: eliminate row 1.
        //   Now col 2, col 3 have weight 1 (only row 2). -> eliminate row 2.
        // All rows eliminated. No dependencies (which is correct: 3 rows, 4 cols,
        // full rank so no dependencies).
        let ncols = 4;
        let mut rows: Vec<BitRow> = (0..3).map(|_| BitRow::new(ncols)).collect();
        rows[0].set(0);
        rows[0].set(1);
        rows[1].set(1);
        rows[1].set(2);
        rows[1].set(3);
        rows[2].set(2);
        rows[2].set(3);

        let deps = find_dependencies_with_preelim(&rows, ncols);
        // Plain GE also finds no dependencies here (full rank).
        let plain_deps = find_dependencies(&rows, ncols);
        assert_eq!(deps.len(), plain_deps.len());
    }

    #[test]
    fn test_preelim_overdetermined() {
        // Same as test_gaussian_elimination_overdetermined but through preelim.
        let ncols = 3;
        let mut rows: Vec<BitRow> = (0..5).map(|_| BitRow::new(ncols)).collect();
        rows[0].set(0);
        rows[1].set(1);
        rows[2].set(2);
        rows[3].set(0);
        rows[3].set(1);
        rows[4].set(1);
        rows[4].set(2);

        let deps = find_dependencies_with_preelim(&rows, ncols);
        assert!(deps.len() >= 2, "Should find at least 2 dependencies");
        for dep in &deps {
            assert!(verify_dependency(&rows, dep, ncols));
        }
    }

    #[test]
    fn test_preelim_empty_input() {
        let deps = find_dependencies_with_preelim(&[], 0);
        assert!(deps.is_empty());
        let deps = find_dependencies_with_preelim(&[], 5);
        assert!(deps.is_empty());
    }

    #[test]
    fn test_preelim_single_row() {
        let ncols = 3;
        let mut rows = vec![BitRow::new(ncols)];
        rows[0].set(0);
        rows[0].set(1);
        let deps = find_dependencies_with_preelim(&rows, ncols);
        // A single row can never form a dependency (need >= 2 rows).
        assert!(deps.is_empty());
    }

    #[test]
    fn test_preelim_identical_rows() {
        // Two identical rows XOR to zero.
        let ncols = 4;
        let mut rows: Vec<BitRow> = (0..2).map(|_| BitRow::new(ncols)).collect();
        rows[0].set(0);
        rows[0].set(2);
        rows[0].set(3);
        rows[1].set(0);
        rows[1].set(2);
        rows[1].set(3);

        let deps = find_dependencies_with_preelim(&rows, ncols);
        assert_eq!(deps.len(), 1);
        assert_eq!(deps[0], vec![0, 1]);
    }

    #[test]
    fn test_preelim_large_sparse_matrix() {
        // Simulate an NFS-like sparse matrix: many singleton columns from
        // large primes, a few weight-2 columns, and a small dense core.
        // 20 rows, 30 columns.
        // Cols 0..9: each appears in exactly 1 row (singletons).
        // Cols 10..14: each appears in exactly 2 rows (weight-2).
        // Cols 15..29: dense core.
        let ncols = 30;
        let nrows = 20;
        let mut rows: Vec<BitRow> = (0..nrows).map(|_| BitRow::new(ncols)).collect();

        // Singleton columns: col c appears only in row c (for c=0..9).
        for c in 0..10 {
            rows[c].set(c);
        }
        // Weight-2 columns: col 10+k appears in rows 2*k and 2*k+1.
        for k in 0..5 {
            rows[2 * k].set(10 + k);
            rows[2 * k + 1].set(10 + k);
        }
        // Dense core: random-ish pattern on cols 15..29 for rows 10..19.
        // Use a simple deterministic pattern.
        for r in 10..20 {
            for c in 15..30 {
                if ((r * 7 + c * 13) % 3) == 0 {
                    rows[r].set(c);
                }
            }
        }

        let deps_preelim = find_dependencies_with_preelim(&rows, ncols);
        let deps_plain = find_dependencies(&rows, ncols);

        // Both should find the same number of dependencies.
        // (Pre-elim doesn't change the null space, just computes it faster.)
        assert_eq!(
            deps_preelim.len(),
            deps_plain.len(),
            "Preelim found {} deps vs plain {}",
            deps_preelim.len(),
            deps_plain.len()
        );

        // Verify all preelim dependencies are valid.
        for dep in &deps_preelim {
            assert!(
                verify_dependency(&rows, dep, ncols),
                "Invalid dependency: {:?}",
                dep
            );
        }
    }

    #[test]
    fn test_preelim_all_zero_rows() {
        // All-zero rows should not produce spurious dependencies.
        let ncols = 3;
        let rows: Vec<BitRow> = (0..3).map(|_| BitRow::new(ncols)).collect();
        let deps = find_dependencies_with_preelim(&rows, ncols);
        // Zero rows XOR to zero, but each individual zero row is "itself"
        // in the history. Any pair of zero rows has XOR = zero, so we
        // should get dependencies. Plain GE does the same.
        let plain = find_dependencies(&rows, ncols);
        assert_eq!(deps.len(), plain.len());
    }

    #[test]
    fn test_preelim_weight2_creates_zero_row() {
        // After a weight-2 merge, the merged row might become all-zero,
        // producing a dependency purely from pre-elimination.
        // Row 0: col {0, 1}
        // Row 1: col {0, 1}
        // Col 0 has weight 2 in rows {0, 1}. Merge: row0 ^= row1 -> row0 = {}.
        // Row 1 eliminated. History of row0 = {0, 1}. Row0 is now zero.
        // -> dependency {0, 1}.
        let ncols = 2;
        let mut rows: Vec<BitRow> = (0..2).map(|_| BitRow::new(ncols)).collect();
        rows[0].set(0);
        rows[0].set(1);
        rows[1].set(0);
        rows[1].set(1);

        let deps = find_dependencies_with_preelim(&rows, ncols);
        assert_eq!(deps.len(), 1);
        assert_eq!(deps[0], vec![0, 1]);
        assert!(verify_dependency(&rows, &deps[0], ncols));
    }
}
