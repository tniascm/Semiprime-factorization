use std::collections::HashSet;

use crate::arith::QuadCharSet;
use crate::types::BitRow;
use rayon::prelude::*;

pub use crate::linalg_bw::{
    find_dependencies_bw, find_dependencies_with_preelim_bw,
    find_dependencies_with_preelim_bw_max,
};

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
    find_dependencies_max(rows, ncols, None)
}

/// Gaussian elimination over GF(2) with optional early exit.
///
/// When `max_deps` is `Some(limit)`, the function stops processing pivot
/// columns once `limit` dependencies have been found, avoiding unnecessary
/// work when only a small number of dependencies are needed (e.g., for sqrt).
pub fn find_dependencies_max(
    rows: &[BitRow],
    ncols: usize,
    max_deps: Option<usize>,
) -> Vec<Vec<usize>> {
    let nrows = rows.len();
    if nrows == 0 || ncols == 0 {
        return vec![];
    }

    let dep_limit = max_deps.unwrap_or(usize::MAX);

    let mut matrix: Vec<BitRow> = rows.to_vec();
    let mut history: Vec<BitRow> = (0..nrows)
        .map(|i| {
            let mut h = BitRow::new(nrows);
            h.set(i);
            h
        })
        .collect();

    let mut pivot_row = 0;
    let mut deps_found = 0usize;
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
            None => {
                // No pivot found for this column — it contributes a dependency.
                deps_found += 1;
                if deps_found >= dep_limit {
                    break;
                }
                continue;
            }
        };

        if pr != pivot_row {
            matrix.swap(pr, pivot_row);
            history.swap(pr, pivot_row);
        }

        // Forward-only elimination: only reduce rows BELOW the pivot.
        let remaining = nrows - pivot_row - 1;
        let pivot_data = matrix[pivot_row].clone();
        let pivot_hist = history[pivot_row].clone();
        if remaining > 128 {
            // Parallel path: use rayon for large remaining row counts.
            let (_, lower_matrix) = matrix.split_at_mut(pivot_row + 1);
            let (_, lower_history) = history.split_at_mut(pivot_row + 1);
            lower_matrix
                .par_iter_mut()
                .zip(lower_history.par_iter_mut())
                .for_each(|(row, hist)| {
                    if row.get(col) {
                        row.xor_with(&pivot_data);
                        hist.xor_with(&pivot_hist);
                    }
                });
        } else {
            for r in (pivot_row + 1)..nrows {
                if matrix[r].get(col) {
                    matrix[r].xor_with(&pivot_data);
                    history[r].xor_with(&pivot_hist);
                }
            }
        }

        pivot_row += 1;
    }

    // Extract dependencies: zero rows whose history has ≥2 original rows.
    // Use word-level bit traversal for O(set_bits) extraction instead of O(nrows).
    let mut deps = Vec::new();
    for r in 0..nrows {
        if matrix[r].is_zero() {
            let mut dep = Vec::new();
            for (wi, &word) in history[r].bits.iter().enumerate() {
                let mut w = word;
                while w != 0 {
                    let bit = w.trailing_zeros() as usize;
                    dep.push(wi * 64 + bit);
                    w &= w - 1;
                }
            }
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

    // Important: never choose all basis vectors every time, otherwise for small
    // nullspaces (n <= k) randomization collapses to one repeated dependency.
    let k_eff = k.clamp(2, n - 1);

    // Parallel generation: split work into chunks with distinct seeds.
    // Each chunk generates candidates independently, then we dedup globally.
    let n_chunks = rayon::current_num_threads().max(1);
    let chunk_size = (n_random + n_chunks - 1) / n_chunks;

    let chunk_results: Vec<Vec<(u64, Vec<usize>)>> = (0..n_chunks)
        .into_par_iter()
        .map(|chunk_idx| {
            let chunk_start = chunk_idx * chunk_size;
            let chunk_end = (chunk_start + chunk_size).min(n_random);
            if chunk_start >= n_random {
                return Vec::new();
            }
            // Derive per-chunk seed from global seed + chunk index
            let mut rng_state = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(chunk_idx as u64 * 0x9e3779b97f4a7c15);

            let mut results = Vec::with_capacity(chunk_end - chunk_start);
            let mut combined = BitRow::new(universe);

            for _ in chunk_start..chunk_end {
                rng_state = rng_state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let target_k = if k_eff > 2 {
                    2 + ((rng_state >> 17) as usize % (k_eff - 1))
                } else {
                    2
                };

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

                for w in combined.bits.iter_mut() {
                    *w = 0;
                }
                for (basis_idx, basis_br) in basis_bits.iter().enumerate() {
                    if chosen_mask.get(basis_idx) {
                        combined.xor_with(basis_br);
                    }
                }

                let popcount: u32 = combined.bits.iter().map(|w| w.count_ones()).sum();
                if popcount >= 3 {
                    let mut h = 0xcbf29ce484222325u64;
                    for &w in &combined.bits {
                        h ^= w;
                        h = h.wrapping_mul(0x100000001b3);
                    }
                    let mut dep = Vec::with_capacity(popcount as usize);
                    for (wi, &word) in combined.bits.iter().enumerate() {
                        let mut w = word;
                        while w != 0 {
                            let bit = w.trailing_zeros();
                            dep.push(wi * 64 + bit as usize);
                            w &= w - 1;
                        }
                    }
                    results.push((h, dep));
                }
            }
            results
        })
        .collect();

    // Merge: dedup across all chunks using hash set
    let mut all_deps = basis_deps.to_vec();
    let mut seen_hashes: HashSet<u64> = HashSet::with_capacity(n + n_random);
    for br in &basis_bits {
        let mut h = 0xcbf29ce484222325u64;
        for &w in &br.bits {
            h ^= w;
            h = h.wrapping_mul(0x100000001b3);
        }
        seen_hashes.insert(h);
    }
    for chunk in chunk_results {
        for (h, dep) in chunk {
            if seen_hashes.insert(h) {
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
    find_dependencies_with_preelim_max(rows, ncols, None)
}

/// Pre-elimination + dense GE with optional early exit on dependency count.
///
/// Behaves identically to [`find_dependencies_with_preelim`] when `max_deps`
/// is `None`. When `Some(limit)`, the inner GE call stops after finding
/// `limit` dependencies (minus any already found during pre-elimination).
pub fn find_dependencies_with_preelim_max(
    rows: &[BitRow],
    ncols: usize,
    max_deps: Option<usize>,
) -> Vec<Vec<usize>> {
    let nrows = rows.len();
    if nrows == 0 || ncols == 0 {
        return vec![];
    }

    let preelim_start = std::time::Instant::now();

    // Work on a mutable copy. Each row in `matrix` also tracks which original
    // rows it represents (via XOR merges). We store this as a Vec<Vec<usize>>
    // since during pre-elimination the merges are few and lists stay short.
    let mut matrix: Vec<BitRow> = rows.to_vec();
    // history[i] = set of original row indices that have been XOR-ed into row i.
    let mut history: Vec<Vec<usize>> = (0..nrows).map(|i| vec![i]).collect();

    let mut row_alive = vec![true; nrows];
    let mut col_alive = vec![true; ncols];

    // Phase 1: Queue-based singleton elimination (O(nnz) total).
    // Compute column weights via word-level bit traversal.
    let mut col_weight: Vec<u32> = vec![0; ncols];
    // Track one row per column for singleton lookup (avoids re-scan).
    let mut col_first_row: Vec<usize> = vec![usize::MAX; ncols];
    for r in 0..nrows {
        if !row_alive[r] {
            continue;
        }
        for (wi, &word) in matrix[r].bits.iter().enumerate() {
            let mut w = word;
            while w != 0 {
                let bit = w.trailing_zeros() as usize;
                let c = wi * 64 + bit;
                if c < ncols && col_alive[c] {
                    col_weight[c] += 1;
                    if col_weight[c] == 1 {
                        col_first_row[c] = r;
                    }
                }
                w &= w - 1;
            }
        }
    }

    // Seed singleton queue.
    let mut singleton_queue: Vec<usize> = Vec::new();
    for c in 0..ncols {
        if col_alive[c] && col_weight[c] == 1 {
            singleton_queue.push(c);
        }
    }

    while let Some(c) = singleton_queue.pop() {
        if !col_alive[c] || col_weight[c] != 1 {
            continue;
        }
        // Find the single row containing this column.
        let r = if col_first_row[c] != usize::MAX
            && row_alive[col_first_row[c]]
            && matrix[col_first_row[c]].get(c)
        {
            col_first_row[c]
        } else {
            // Cached row was killed; scan for the actual singleton row.
            let mut found = usize::MAX;
            for ri in 0..nrows {
                if row_alive[ri] && matrix[ri].get(c) {
                    found = ri;
                    break;
                }
            }
            found
        };
        if r == usize::MAX {
            col_alive[c] = false;
            continue;
        }

        // Remove this row; decrement weights for all its other columns.
        row_alive[r] = false;
        col_alive[c] = false;
        for (wi, &word) in matrix[r].bits.iter().enumerate() {
            let mut w = word;
            while w != 0 {
                let bit = w.trailing_zeros() as usize;
                let c2 = wi * 64 + bit;
                if c2 < ncols && col_alive[c2] {
                    col_weight[c2] = col_weight[c2].saturating_sub(1);
                    if col_weight[c2] == 1 {
                        singleton_queue.push(c2);
                    }
                }
                w &= w - 1;
            }
        }
    }

    let phase1_ms = preelim_start.elapsed().as_secs_f64() * 1000.0;
    let phase1_alive = (0..nrows).filter(|&r| row_alive[r]).count();
    let phase2_start = std::time::Instant::now();

    // Phase 2: Batched weight-2 column merges with integrated singleton queue.
    // Each pass: scan alive rows once to find weight-2 columns and their rows,
    // batch non-conflicting merges, process resulting singletons via queue,
    // then repeat until convergence.
    let mut col_r1: Vec<usize> = vec![usize::MAX; ncols];
    let mut col_r2: Vec<usize> = vec![usize::MAX; ncols];
    let mut row_used = vec![false; nrows];
    let mut merges: Vec<(usize, usize, usize)> = Vec::new();
    loop {
        // Scan all alive rows to build weight-2 column→rows mapping.
        for c in 0..ncols {
            if col_alive[c] {
                col_weight[c] = 0;
                col_r1[c] = usize::MAX;
                col_r2[c] = usize::MAX;
            }
        }
        for r in 0..nrows {
            if !row_alive[r] {
                continue;
            }
            for (wi, &word) in matrix[r].bits.iter().enumerate() {
                let mut w = word;
                while w != 0 {
                    let bit = w.trailing_zeros() as usize;
                    let c = wi * 64 + bit;
                    if c < ncols && col_alive[c] {
                        col_weight[c] += 1;
                        if col_weight[c] == 1 {
                            col_r1[c] = r;
                        } else if col_weight[c] == 2 {
                            col_r2[c] = r;
                        }
                    }
                    w &= w - 1;
                }
            }
        }

        // Collect weight-2 merge candidates, skipping row conflicts.
        // Process singletons via queue (same as Phase 1) instead of re-scanning.
        merges.clear();
        for i in 0..nrows {
            row_used[i] = false;
        }
        let mut singleton_queue2: Vec<usize> = Vec::new();
        for c in 0..ncols {
            if !col_alive[c] {
                continue;
            }
            match col_weight[c] {
                0 => {
                    col_alive[c] = false;
                }
                1 => {
                    singleton_queue2.push(c);
                }
                2 => {
                    let r1 = col_r1[c];
                    let r2 = col_r2[c];
                    if r1 != usize::MAX && r2 != usize::MAX && !row_used[r1] && !row_used[r2] {
                        merges.push((c, r1, r2));
                        row_used[r1] = true;
                        row_used[r2] = true;
                    }
                }
                _ => {}
            }
        }

        // Process singletons via queue (O(removed × avg_row_weight)).
        while let Some(c) = singleton_queue2.pop() {
            if !col_alive[c] || col_weight[c] != 1 {
                continue;
            }
            let r = if col_r1[c] != usize::MAX && row_alive[col_r1[c]] && matrix[col_r1[c]].get(c) {
                col_r1[c]
            } else {
                let mut found = usize::MAX;
                for ri in 0..nrows {
                    if row_alive[ri] && matrix[ri].get(c) {
                        found = ri;
                        break;
                    }
                }
                found
            };
            if r == usize::MAX {
                col_alive[c] = false;
                continue;
            }
            row_alive[r] = false;
            col_alive[c] = false;
            for (wi, &word) in matrix[r].bits.iter().enumerate() {
                let mut w = word;
                while w != 0 {
                    let bit = w.trailing_zeros() as usize;
                    let c2 = wi * 64 + bit;
                    if c2 < ncols && col_alive[c2] {
                        col_weight[c2] = col_weight[c2].saturating_sub(1);
                        if col_weight[c2] == 1 {
                            singleton_queue2.push(c2);
                        }
                    }
                    w &= w - 1;
                }
            }
        }

        if merges.is_empty() {
            break;
        }

        // Execute all batched merges.
        for &(c, r1, r2) in &merges {
            let r2_data = matrix[r2].clone();
            matrix[r1].xor_with(&r2_data);
            // Merge history: symmetric difference using HashSet for O(n+m).
            let h1 = std::mem::take(&mut history[r1]);
            let h2 = &history[r2];
            let mut set: std::collections::HashSet<usize> =
                h1.into_iter().collect();
            for &idx in h2 {
                if !set.remove(&idx) {
                    set.insert(idx);
                }
            }
            history[r1] = set.into_iter().collect();
            row_alive[r2] = false;
            col_alive[c] = false;
        }
    }

    let phase2_ms = phase2_start.elapsed().as_secs_f64() * 1000.0;
    let phase2_alive = (0..nrows).filter(|&r| row_alive[r]).count();

    // Phase 3: Weight-3 column merges with Markowitz pivot selection.
    // For each weight-3 column, XOR the lightest row (by total weight) into
    // the other two, eliminating one row and one column. Followed by singleton
    // and weight-2 cleanup. Repeat until no weight-3 columns remain or fill-in
    // exceeds threshold.
    let phase3_start = std::time::Instant::now();
    let phase3_max_fill = std::env::var("RUST_NFS_PREELIM_MAXFILL")
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(2.0); // stop when total nnz exceeds 2x original
    let phase3_max_weight: usize = std::env::var("RUST_NFS_PREELIM_MAXWEIGHT")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(50);

    // Compute initial nnz (number of non-zero entries in alive rows/cols).
    let initial_nnz: usize = (0..nrows)
        .filter(|&r| row_alive[r])
        .map(|r| {
            matrix[r].bits.iter().enumerate().map(|(wi, &word)| {
                let mut w = word;
                let mut cnt = 0usize;
                while w != 0 {
                    let bit = w.trailing_zeros() as usize;
                    let c = wi * 64 + bit;
                    if c < ncols && col_alive[c] {
                        cnt += 1;
                    }
                    w &= w - 1;
                }
                cnt
            }).sum::<usize>()
        })
        .sum();
    let nnz_limit = ((initial_nnz as f64) * phase3_max_fill) as usize;

    // Row weight cache for Markowitz selection.
    let mut row_weight: Vec<usize> = vec![0; nrows];
    for r in 0..nrows {
        if !row_alive[r] { continue; }
        row_weight[r] = matrix[r].bits.iter().enumerate().map(|(wi, &word)| {
            let mut w = word;
            let mut cnt = 0usize;
            while w != 0 {
                let bit = w.trailing_zeros() as usize;
                let c = wi * 64 + bit;
                if c < ncols && col_alive[c] { cnt += 1; }
                w &= w - 1;
            }
            cnt
        }).sum();
    }

    let mut current_nnz = initial_nnz;
    let mut phase3_merges = 0usize;
    let mut col_r3: Vec<usize> = vec![usize::MAX; ncols];

    'phase3: loop {
        // Recompute column weights and row associations for weight-3 detection.
        for c in 0..ncols {
            if col_alive[c] {
                col_weight[c] = 0;
                col_r1[c] = usize::MAX;
                col_r2[c] = usize::MAX;
                col_r3[c] = usize::MAX;
            }
        }
        for r in 0..nrows {
            if !row_alive[r] { continue; }
            for (wi, &word) in matrix[r].bits.iter().enumerate() {
                let mut w = word;
                while w != 0 {
                    let bit = w.trailing_zeros() as usize;
                    let c = wi * 64 + bit;
                    if c < ncols && col_alive[c] {
                        col_weight[c] += 1;
                        match col_weight[c] {
                            1 => col_r1[c] = r,
                            2 => col_r2[c] = r,
                            3 => col_r3[c] = r,
                            _ => {}
                        }
                    }
                    w &= w - 1;
                }
            }
        }

        // Collect weight-3 merge candidates sorted by minimum row weight (Markowitz).
        merges.clear();
        for i in 0..nrows { row_used[i] = false; }
        let mut singleton_queue3: Vec<usize> = Vec::new();

        // First handle singletons and weight-2 from this scan.
        for c in 0..ncols {
            if !col_alive[c] { continue; }
            match col_weight[c] {
                0 => { col_alive[c] = false; }
                1 => { singleton_queue3.push(c); }
                _ => {}
            }
        }

        // Process singletons.
        while let Some(c) = singleton_queue3.pop() {
            if !col_alive[c] || col_weight[c] != 1 { continue; }
            let r = if col_r1[c] != usize::MAX && row_alive[col_r1[c]] && matrix[col_r1[c]].get(c) {
                col_r1[c]
            } else {
                let mut found = usize::MAX;
                for ri in 0..nrows {
                    if row_alive[ri] && matrix[ri].get(c) { found = ri; break; }
                }
                found
            };
            if r == usize::MAX { col_alive[c] = false; continue; }
            current_nnz = current_nnz.saturating_sub(row_weight[r]);
            row_alive[r] = false;
            col_alive[c] = false;
            for (wi, &word) in matrix[r].bits.iter().enumerate() {
                let mut w = word;
                while w != 0 {
                    let bit = w.trailing_zeros() as usize;
                    let c2 = wi * 64 + bit;
                    if c2 < ncols && col_alive[c2] {
                        col_weight[c2] = col_weight[c2].saturating_sub(1);
                        if col_weight[c2] == 1 { singleton_queue3.push(c2); }
                    }
                    w &= w - 1;
                }
            }
        }

        // Now collect weight-3 candidates (after singleton cleanup).
        // Sort by pivot weight (Markowitz: lightest pivot → least fill-in).
        struct W3Candidate { col: usize, pivot: usize, r_a: usize, r_b: usize, pivot_w: usize }
        let mut candidates: Vec<W3Candidate> = Vec::new();
        for c in 0..ncols {
            if !col_alive[c] || col_weight[c] != 3 { continue; }
            let r1 = col_r1[c]; let r2 = col_r2[c]; let r3 = col_r3[c];
            if r1 == usize::MAX || r2 == usize::MAX || r3 == usize::MAX { continue; }
            if !row_alive[r1] || !row_alive[r2] || !row_alive[r3] { continue; }
            // Pick lightest row as pivot (Markowitz criterion).
            let w1 = row_weight[r1]; let w2 = row_weight[r2]; let w3 = row_weight[r3];
            let (pivot, ra, rb, pw) = if w1 <= w2 && w1 <= w3 {
                (r1, r2, r3, w1)
            } else if w2 <= w3 {
                (r2, r1, r3, w2)
            } else {
                (r3, r1, r2, w3)
            };
            // Skip candidates where pivot is too heavy (limits fill-in and
            // keeps dependency histories short for sqrt quality).
            if pw > phase3_max_weight { continue; }
            candidates.push(W3Candidate { col: c, pivot, r_a: ra, r_b: rb, pivot_w: pw });
        }

        if candidates.is_empty() { break 'phase3; }

        // Sort by pivot weight ascending for best Markowitz ordering.
        candidates.sort_unstable_by_key(|c| c.pivot_w);

        // Batch non-conflicting merges (no row used twice).
        let mut batch_count = 0usize;
        for cand in &candidates {
            if row_used[cand.pivot] || row_used[cand.r_a] || row_used[cand.r_b] { continue; }
            // Estimate fill-in: up to 2*(pivot_weight-1) new entries.
            let est_fill = 2 * cand.pivot_w.saturating_sub(1);
            if current_nnz + est_fill > nnz_limit { break 'phase3; }

            // XOR pivot into r_a and r_b, then remove pivot row and column.
            let pivot_data = matrix[cand.pivot].clone();
            let pivot_hist = std::mem::take(&mut history[cand.pivot]);

            // Merge into r_a.
            matrix[cand.r_a].xor_with(&pivot_data);
            {
                let h_a = std::mem::take(&mut history[cand.r_a]);
                let mut set: std::collections::HashSet<usize> = h_a.into_iter().collect();
                for &idx in &pivot_hist { if !set.remove(&idx) { set.insert(idx); } }
                history[cand.r_a] = set.into_iter().collect();
            }
            // Recompute row weight for r_a.
            let new_w_a: usize = matrix[cand.r_a].bits.iter().enumerate().map(|(wi, &word)| {
                let mut w = word; let mut cnt = 0;
                while w != 0 { let bit = w.trailing_zeros() as usize; let c = wi*64+bit; if c<ncols && col_alive[c] { cnt+=1; } w &= w-1; }
                cnt
            }).sum();
            current_nnz = current_nnz + new_w_a - row_weight[cand.r_a];
            row_weight[cand.r_a] = new_w_a;

            // Merge into r_b.
            matrix[cand.r_b].xor_with(&pivot_data);
            {
                let h_b = std::mem::take(&mut history[cand.r_b]);
                let mut set: std::collections::HashSet<usize> = h_b.into_iter().collect();
                for &idx in &pivot_hist { if !set.remove(&idx) { set.insert(idx); } }
                history[cand.r_b] = set.into_iter().collect();
            }
            let new_w_b: usize = matrix[cand.r_b].bits.iter().enumerate().map(|(wi, &word)| {
                let mut w = word; let mut cnt = 0;
                while w != 0 { let bit = w.trailing_zeros() as usize; let c = wi*64+bit; if c<ncols && col_alive[c] { cnt+=1; } w &= w-1; }
                cnt
            }).sum();
            current_nnz = current_nnz + new_w_b - row_weight[cand.r_b];
            row_weight[cand.r_b] = new_w_b;

            // Remove pivot row and column.
            current_nnz = current_nnz.saturating_sub(row_weight[cand.pivot]);
            row_alive[cand.pivot] = false;
            col_alive[cand.col] = false;
            row_used[cand.pivot] = true;
            row_used[cand.r_a] = true;
            row_used[cand.r_b] = true;
            batch_count += 1;
            phase3_merges += 1;
        }

        if batch_count == 0 { break 'phase3; }
    }

    let phase3_ms = phase3_start.elapsed().as_secs_f64() * 1000.0;
    let phase3_alive = (0..nrows).filter(|&r| row_alive[r]).count();

    // Phase 4: Generalized weight-K column merges (K = 3..max_col_weight).
    // For each column with weight K, pick the lightest row as Markowitz pivot,
    // XOR into the other K-1 rows. Lower-weight columns are processed first.
    let phase4_start = std::time::Instant::now();
    let phase4_max_col_weight: usize = std::env::var("RUST_NFS_PREELIM_MAXCOLWEIGHT")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(10);
    let mut phase4_merges = 0usize;
    // Flat storage for column→rows index: col_rows_flat[col_rows_offset[c]..] holds
    // up to phase4_max_col_weight row indices for column c.
    let mut col_rows_flat: Vec<usize> = Vec::new();
    let mut col_rows_offset: Vec<usize> = vec![0; ncols];

    'phase4: loop {
        // Recompute column weights AND build column→rows index in one pass.
        col_rows_flat.clear();
        for c in 0..ncols {
            if col_alive[c] {
                col_weight[c] = 0;
                col_rows_offset[c] = 0;
            }
        }
        // First pass: count weights.
        for r in 0..nrows {
            if !row_alive[r] { continue; }
            for (wi, &word) in matrix[r].bits.iter().enumerate() {
                let mut w = word;
                while w != 0 {
                    let bit = w.trailing_zeros() as usize;
                    let c = wi * 64 + bit;
                    if c < ncols && col_alive[c] {
                        col_weight[c] += 1;
                    }
                    w &= w - 1;
                }
            }
        }
        // Compute offsets for columns we'll track (weight 1..=max_col_weight).
        let mut total_slots = 0usize;
        for c in 0..ncols {
            if col_alive[c] && col_weight[c] >= 1 && (col_weight[c] as usize) <= phase4_max_col_weight {
                col_rows_offset[c] = total_slots;
                total_slots += col_weight[c] as usize;
            } else {
                col_rows_offset[c] = usize::MAX;
            }
        }
        col_rows_flat.resize(total_slots, usize::MAX);
        // Track how many rows we've recorded per column.
        let mut col_fill: Vec<u32> = vec![0; ncols];
        // Second pass: record row indices.
        for r in 0..nrows {
            if !row_alive[r] { continue; }
            for (wi, &word) in matrix[r].bits.iter().enumerate() {
                let mut w = word;
                while w != 0 {
                    let bit = w.trailing_zeros() as usize;
                    let c = wi * 64 + bit;
                    if c < ncols && col_alive[c] && col_rows_offset[c] != usize::MAX {
                        let idx = col_rows_offset[c] + col_fill[c] as usize;
                        if idx < col_rows_flat.len() {
                            col_rows_flat[idx] = r;
                            col_fill[c] += 1;
                        }
                    }
                    w &= w - 1;
                }
            }
        }

        // Clean up singletons and dead columns.
        let mut singleton_queue4: Vec<usize> = Vec::new();
        for c in 0..ncols {
            if !col_alive[c] { continue; }
            match col_weight[c] {
                0 => { col_alive[c] = false; }
                1 => { singleton_queue4.push(c); }
                _ => {}
            }
        }
        while let Some(c) = singleton_queue4.pop() {
            if !col_alive[c] || col_weight[c] != 1 { continue; }
            // Find the single row for this column from the index.
            let r = if col_rows_offset[c] != usize::MAX {
                let ri = col_rows_flat[col_rows_offset[c]];
                if ri != usize::MAX && row_alive[ri] && matrix[ri].get(c) { ri } else { usize::MAX }
            } else {
                usize::MAX
            };
            // Fallback scan if cached row was killed.
            let r = if r == usize::MAX {
                let mut found = usize::MAX;
                for ri in 0..nrows {
                    if row_alive[ri] && matrix[ri].get(c) { found = ri; break; }
                }
                found
            } else { r };
            if r == usize::MAX { col_alive[c] = false; continue; }
            current_nnz = current_nnz.saturating_sub(row_weight[r]);
            row_alive[r] = false;
            col_alive[c] = false;
            for (wi, &word) in matrix[r].bits.iter().enumerate() {
                let mut w = word;
                while w != 0 {
                    let bit = w.trailing_zeros() as usize;
                    let c2 = wi * 64 + bit;
                    if c2 < ncols && col_alive[c2] {
                        col_weight[c2] = col_weight[c2].saturating_sub(1);
                        if col_weight[c2] == 1 { singleton_queue4.push(c2); }
                    }
                    w &= w - 1;
                }
            }
        }

        // Collect merge candidates for columns with weight 3..=max_col_weight
        // using pre-built column→rows index.
        struct WkCandidate { col: usize, pivot: usize, targets: Vec<usize>, pivot_w: usize }
        let mut candidates: Vec<WkCandidate> = Vec::new();
        for i in 0..nrows { row_used[i] = false; }

        for c in 0..ncols {
            if !col_alive[c] { continue; }
            let cw = col_weight[c] as usize;
            if cw < 3 || cw > phase4_max_col_weight { continue; }
            if col_rows_offset[c] == usize::MAX { continue; }

            // Read rows from the pre-built index.
            let offset = col_rows_offset[c];
            let mut col_rows: Vec<(usize, usize)> = Vec::with_capacity(cw);
            for k in 0..cw {
                let ri = col_rows_flat[offset + k];
                if ri == usize::MAX || !row_alive[ri] { break; }
                col_rows.push((ri, row_weight[ri]));
            }
            if col_rows.len() != cw { continue; }

            // Pick lightest row as pivot (Markowitz criterion).
            let (pivot_idx, &(pivot, pw)) = col_rows.iter().enumerate()
                .min_by_key(|(_, (_, w))| *w).unwrap();
            if pw > phase3_max_weight { continue; }

            let targets: Vec<usize> = col_rows.iter().enumerate()
                .filter(|(i, _)| *i != pivot_idx)
                .map(|(_, (r, _))| *r)
                .collect();
            candidates.push(WkCandidate { col: c, pivot, targets, pivot_w: pw });
        }

        if candidates.is_empty() { break 'phase4; }

        // Sort by pivot weight ascending.
        candidates.sort_unstable_by_key(|c| c.pivot_w);

        let mut batch_count = 0usize;
        for cand in &candidates {
            if row_used[cand.pivot] || cand.targets.iter().any(|&t| row_used[t]) { continue; }
            let n_targets = cand.targets.len();
            let est_fill = n_targets * cand.pivot_w.saturating_sub(1);
            if current_nnz + est_fill > nnz_limit { break 'phase4; }

            let pivot_data = matrix[cand.pivot].clone();
            let pivot_hist = std::mem::take(&mut history[cand.pivot]);

            for &target in &cand.targets {
                matrix[target].xor_with(&pivot_data);
                let h_t = std::mem::take(&mut history[target]);
                let mut set: HashSet<usize> = h_t.into_iter().collect();
                for &idx in &pivot_hist { if !set.remove(&idx) { set.insert(idx); } }
                history[target] = set.into_iter().collect();
                let new_w: usize = matrix[target].bits.iter().enumerate().map(|(wi, &word)| {
                    let mut w = word; let mut cnt = 0;
                    while w != 0 { let bit = w.trailing_zeros() as usize; let c = wi*64+bit; if c<ncols && col_alive[c] { cnt+=1; } w &= w-1; }
                    cnt
                }).sum();
                current_nnz = current_nnz + new_w - row_weight[target];
                row_weight[target] = new_w;
            }

            current_nnz = current_nnz.saturating_sub(row_weight[cand.pivot]);
            row_alive[cand.pivot] = false;
            col_alive[cand.col] = false;
            row_used[cand.pivot] = true;
            for &t in &cand.targets { row_used[t] = true; }
            batch_count += 1;
            phase4_merges += 1;
        }

        if batch_count == 0 { break 'phase4; }
    }

    let phase4_ms = phase4_start.elapsed().as_secs_f64() * 1000.0;
    let phase4_alive = (0..nrows).filter(|&r| row_alive[r]).count();
    eprintln!(
        "  pre-elim: phase1={:.0}ms ({}->{} rows), phase2={:.0}ms ({}->{} rows), phase3={:.0}ms ({}->{} rows, {} w3-merges), phase4={:.0}ms ({}->{} rows, {} wK-merges, nnz {}/{})",
        phase1_ms, nrows, phase1_alive, phase2_ms, phase1_alive, phase2_alive,
        phase3_ms, phase2_alive, phase3_alive, phase3_merges,
        phase4_ms, phase3_alive, phase4_alive, phase4_merges, current_nnz, nnz_limit
    );

    let compact_start = std::time::Instant::now();
    // Build compacted matrix from surviving rows and columns.
    let surviving_rows: Vec<usize> = (0..nrows).filter(|&r| row_alive[r]).collect();
    let surviving_cols: Vec<usize> = (0..ncols).filter(|&c| col_alive[c]).collect();

    // Check for dependencies among rows that became all-zero after pre-elimination,
    // and partition surviving rows into zero-rows (dependencies) and non-zero (for GE).
    // Build a column-alive bitmask for fast zero-check via word-level AND.
    let nwords = (ncols + 63) / 64;
    let mut col_alive_mask = vec![0u64; nwords];
    for &c in &surviving_cols {
        col_alive_mask[c / 64] |= 1u64 << (c % 64);
    }

    let mut deps = Vec::new();
    let mut ge_rows = Vec::new();
    for &r in &surviving_rows {
        // Check if this row is zero across all surviving columns using word-level AND.
        let all_zero = matrix[r]
            .bits
            .iter()
            .zip(col_alive_mask.iter())
            .all(|(&row_word, &mask_word)| row_word & mask_word == 0);
        if all_zero {
            let mut dep = history[r].clone();
            dep.sort_unstable();
            if dep.len() >= 2 {
                deps.push(dep);
            }
        } else {
            ge_rows.push(r);
        }
    }

    if ge_rows.is_empty() || surviving_cols.is_empty() {
        return deps;
    }

    let compact_nrows = ge_rows.len();
    let compact_ncols = surviving_cols.len();

    // Build old-to-new column index map for fast lookup during compaction.
    let mut old_to_new_col = vec![usize::MAX; ncols];
    for (new_c, &old_c) in surviving_cols.iter().enumerate() {
        old_to_new_col[old_c] = new_c;
    }

    let mut compact_matrix: Vec<BitRow> = Vec::with_capacity(compact_nrows);
    for &r in &ge_rows {
        let mut compact_row = BitRow::new(compact_ncols);
        // Use word-level bit traversal to find set bits efficiently
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
                w &= w - 1; // clear lowest set bit
            }
        }
        compact_matrix.push(compact_row);
    }

    let compact_ms = compact_start.elapsed().as_secs_f64() * 1000.0;
    eprintln!("  pre-elim: compact {:.0}ms ({} x {}, ge_rows={}, zero_deps={})",
        compact_ms, compact_nrows, compact_ncols, ge_rows.len(), deps.len());

    // Run dense GE on the compacted matrix, subtracting pre-elim deps from budget.
    let ge_inner_start = std::time::Instant::now();
    let remaining_budget = max_deps.map(|limit| limit.saturating_sub(deps.len()));
    let compact_deps = find_dependencies_max(&compact_matrix, compact_ncols, remaining_budget);
    let ge_inner_ms = ge_inner_start.elapsed().as_secs_f64() * 1000.0;
    eprintln!("  pre-elim: GE {:.0}ms -> {} deps from {} rows", ge_inner_ms, compact_deps.len(), compact_nrows);

    // Map compact row indices back to original row indices via history.
    // Pre-convert Vec<usize> histories to BitRow for O(nwords) XOR combining
    // instead of O(history_len) HashSet operations per dependency.
    let map_start = std::time::Instant::now();
    let ge_histories_br: Vec<BitRow> = ge_rows
        .iter()
        .map(|&r| {
            let mut br = BitRow::new(nrows);
            for &idx in &history[r] {
                br.flip(idx);
            }
            br
        })
        .collect();

    let mapped_deps: Vec<Vec<usize>> = compact_deps
        .par_iter()
        .filter_map(|cdep| {
            let mut combined = BitRow::new(nrows);
            for &ci in cdep {
                combined.xor_with(&ge_histories_br[ci]);
            }
            // Extract set bits via word-level traversal.
            let mut dep = Vec::new();
            for (wi, &word) in combined.bits.iter().enumerate() {
                let mut w = word;
                while w != 0 {
                    let bit = w.trailing_zeros() as usize;
                    dep.push(wi * 64 + bit);
                    w &= w - 1;
                }
            }
            if dep.len() >= 2 {
                Some(dep)
            } else {
                None
            }
        })
        .collect();
    deps.extend(mapped_deps);

    let map_ms = map_start.elapsed().as_secs_f64() * 1000.0;
    eprintln!(
        "  pre-elim: map-back {:.0}ms -> {} total deps",
        map_ms,
        deps.len()
    );

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
