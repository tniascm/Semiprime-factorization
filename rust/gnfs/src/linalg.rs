use crate::arith::QuadCharSet;
use crate::types::BitRow;

/// Build the GF(2) exponent matrix from relations.
///
/// Each row corresponds to a relation. Columns are:
/// - Column 0: sign bit (rational side)
/// - Columns 1..1+rat_fb_size: rational factor base exponents mod 2
/// - Column 1+rat_fb_size: sign bit (algebraic side)
/// - Columns 2+rat_fb_size..2+rat_fb_size+alg_pair_count: algebraic (prime,root)
///   pair exponents mod 2 — one column per degree-1 prime ideal
/// - Remaining columns: quadratic character Legendre symbols
///
/// Per-(prime,root) tracking ensures each degree-1 prime ideal has even exponent.
/// Only relations fully explained by degree-1 ideals are accepted (no HD ideal columns).
/// Quadratic characters handle remaining unit/class group constraints.
pub fn build_matrix(
    relations: &[crate::types::Relation],
    rat_fb_size: usize,
    alg_pair_count: usize,
    quad_chars: &QuadCharSet,
) -> (Vec<BitRow>, usize) {
    let n_qc = quad_chars.primes.len();
    let ncols = 2 + rat_fb_size + alg_pair_count + n_qc;
    let mut rows = Vec::with_capacity(relations.len());

    for rel in relations {
        let mut row = BitRow::new(ncols);

        if rel.rational_sign_negative {
            row.set(0);
        }

        for &(idx, exp) in &rel.rational_factors {
            if exp % 2 == 1 {
                row.set(1 + idx as usize);
            }
        }

        if rel.algebraic_sign_negative {
            row.set(1 + rat_fb_size);
        }

        // algebraic_factors stores (flat_pair_index, exp) where
        // flat_pair_index uniquely identifies a (prime, root) pair
        for &(pair_idx, exp) in &rel.algebraic_factors {
            if exp % 2 == 1 {
                row.set(2 + rat_fb_size + pair_idx as usize);
            }
        }

        // Quadratic characters: Legendre symbol ((a - b*r) / q)
        for (i, (&q, &r)) in quad_chars.primes.iter().zip(quad_chars.roots.iter()).enumerate() {
            let q_i = q as i128;
            let val = (rel.a as i128 - rel.b as i128 * r as i128).rem_euclid(q_i) as u64;
            if val == 0 {
                continue; // q divides (a - b*r), treat as +1
            }
            let ls = crate::arith::legendre_symbol(val, q);
            if ls == q - 1 {
                // Legendre symbol is -1 → set bit (odd in GF(2))
                row.set(2 + rat_fb_size + alg_pair_count + i);
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

        for r in 0..nrows {
            if r != pivot_row && matrix[r].get(col) {
                let pivot_data = matrix[pivot_row].clone();
                matrix[r].xor_with(&pivot_data);
                let pivot_hist = history[pivot_row].clone();
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
    use std::collections::HashSet;

    let n = basis_deps.len();
    if n < 2 || k < 2 {
        return basis_deps.to_vec();
    }

    let mut all_deps = basis_deps.to_vec();
    let mut rng_state = seed;

    for _ in 0..n_random {
        // Simple LCG for deterministic PRNG
        let mut chosen = HashSet::new();
        while chosen.len() < k.min(n) {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let idx = (rng_state >> 33) as usize % n;
            chosen.insert(idx);
        }

        // XOR of the chosen basis vectors: symmetric difference of relation indices
        let mut combined: HashSet<usize> = HashSet::new();
        for &basis_idx in &chosen {
            for &rel_idx in &basis_deps[basis_idx] {
                if !combined.remove(&rel_idx) {
                    combined.insert(rel_idx);
                }
            }
        }

        if combined.len() >= 3 {
            let mut dep: Vec<usize> = combined.into_iter().collect();
            dep.sort_unstable();
            all_deps.push(dep);
        }
    }

    all_deps
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
        let mut rows = vec![
            BitRow::new(ncols),
            BitRow::new(ncols),
            BitRow::new(ncols),
        ];
        rows[0].set(0); rows[0].set(2);
        rows[1].set(1); rows[1].set(2);
        rows[2].set(0); rows[2].set(1);

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
        rows[3].set(0); rows[3].set(1);
        rows[4].set(1); rows[4].set(2);

        let deps = find_dependencies(&rows, ncols);
        assert!(deps.len() >= 2, "Should find at least 2 dependencies");
    }
}
