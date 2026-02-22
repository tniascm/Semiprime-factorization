//! Strategy 3: CRT (Chinese Remainder Theorem) Decomposition
//!
//! Decompose N into independent residue components modulo a set of coprime moduli.
//! Once the CRT representation is computed, checking divisibility by any of the
//! moduli or their products is O(1). For other divisors, the computation is
//! independent per divisor and parallelizable.
//!
//! The SSD connection: the CRT decomposition is itself a "parallel form" --
//! instead of a single sequential mod operation, we have k independent residues
//! that can be computed simultaneously and then combined.

use rayon::prelude::*;

use crate::SsdFormulation;

/// CRT Decomposition strategy.
pub struct CrtParallel {
    /// Coprime moduli for the CRT representation.
    pub moduli: Vec<u64>,
}

impl CrtParallel {
    /// Create with first 10 primes as moduli.
    pub fn new_default() -> Self {
        Self {
            moduli: vec![2, 3, 5, 7, 11, 13, 17, 19, 23, 29],
        }
    }
}

/// Extended GCD: returns (gcd, x, y) such that a*x + b*y = gcd(a, b).
pub fn extended_gcd(a: i64, b: i64) -> (i64, i64, i64) {
    if a == 0 {
        return (b, 0, 1);
    }
    let (g, x1, y1) = extended_gcd(b % a, a);
    let x = y1 - (b / a) * x1;
    let y = x1;
    (g, x, y)
}

/// Reconstruct a number from its CRT residues and moduli.
///
/// Given residues[i] = N mod moduli[i] for coprime moduli,
/// reconstructs N mod (product of all moduli).
pub fn crt_reconstruct(residues: &[u64], moduli: &[u64]) -> u64 {
    assert_eq!(
        residues.len(),
        moduli.len(),
        "Residues and moduli must have the same length"
    );

    if residues.is_empty() {
        return 0;
    }

    // Use i128 to avoid overflow during reconstruction
    let product: i128 = moduli.iter().map(|&m| m as i128).product();

    let mut result: i128 = 0;
    for i in 0..residues.len() {
        let mi = moduli[i] as i128;
        let ni = product / mi; // product of all moduli except moduli[i]

        // Find ni_inv such that ni * ni_inv === 1 (mod mi)
        let (_, x, _) = extended_gcd(ni.rem_euclid(mi) as i64, mi as i64);
        let ni_inv = (x as i128).rem_euclid(mi);

        result = (result + residues[i] as i128 * ni % product * ni_inv % product) % product;
    }

    result.rem_euclid(product) as u64
}

impl SsdFormulation for CrtParallel {
    fn name(&self) -> &str {
        "CRT Decomposition"
    }

    fn dimensionality(&self, _n: u64) -> usize {
        self.moduli.len()
    }

    fn parallel(&self, n: u64, divisors: &[u64]) -> Vec<u64> {
        // Step 1: Compute CRT representation of n
        let crt_residues: Vec<u64> = self.moduli.iter().map(|&m| n % m).collect();

        // Step 2: For each divisor d, compute n mod d from the CRT representation
        divisors
            .par_iter()
            .map(|&d| {
                if d <= 1 {
                    return 0;
                }

                // Check if d is one of the CRT moduli -- direct lookup
                if let Some(idx) = self.moduli.iter().position(|&m| m == d) {
                    return crt_residues[idx];
                }

                // Check if d is a product of some subset of the CRT moduli.
                // If so, reconstruct from those components via CRT.
                let mut remaining = d;
                let mut component_indices: Vec<usize> = Vec::new();
                for (i, &m) in self.moduli.iter().enumerate() {
                    if remaining % m == 0 {
                        component_indices.push(i);
                        remaining /= m;
                        // Check if m^2 divides d -- if so, CRT subset won't work
                        // because we only have n mod m, not n mod m^k
                        if d / m % m == 0 {
                            remaining = d; // signal that we can't use CRT
                            component_indices.clear();
                            break;
                        }
                    }
                }

                if remaining == 1 && !component_indices.is_empty() {
                    // d is a squarefree product of some CRT moduli
                    let sub_residues: Vec<u64> =
                        component_indices.iter().map(|&i| crt_residues[i]).collect();
                    let sub_moduli: Vec<u64> =
                        component_indices.iter().map(|&i| self.moduli[i]).collect();
                    let reconstructed = crt_reconstruct(&sub_residues, &sub_moduli);
                    return reconstructed % d;
                }

                // For divisors that are not products of CRT moduli,
                // reconstruct n from CRT and then compute mod d.
                // This is the fallback -- still correct but no speedup from CRT.
                let full_n = crt_reconstruct(&crt_residues, &self.moduli);

                // The CRT reconstruction gives n mod M where M = product of all moduli.
                // If n < M, this is exact. Otherwise we need the original n.
                // Since we're working with u64, check if reconstruction matches.
                let product: u64 = self.moduli.iter().product();
                if n < product {
                    full_n % d
                } else {
                    // Fallback to direct computation for large n
                    n % d
                }
            })
            .collect()
    }
}
