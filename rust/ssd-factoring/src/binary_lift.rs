//! Strategy 1: Binary Indicator Lifting
//!
//! Represent N as a binary vector of its bits. Divisibility by d can be checked
//! by examining specific bit patterns. We construct a matrix M where each row
//! corresponds to a divisor d, and M * bits(N) gives the remainder N mod d
//! (computed via positional values of bits mod d).
//!
//! The SSD connection: the sequential recurrence h_t = A * h_{t-1} + B * x_t
//! that accumulates bit contributions one at a time is replaced by a single
//! matrix-vector product y = M * x where x is the bit vector of N.

use rayon::prelude::*;

use crate::SsdFormulation;

/// Binary Indicator Lifting strategy.
///
/// The matrix M has structure: M[d_index][bit_index] = 2^bit_index mod d.
/// The result for divisor d is: sum(bits[i] * M[d_index][i]) mod d.
pub struct BinaryLift;

impl BinaryLift {
    /// Extract bits of n into a vector where bits[i] = (n >> i) & 1.
    fn extract_bits(n: u64) -> Vec<u64> {
        let num_bits = Self::bit_count(n);
        (0..num_bits).map(|i| (n >> i) & 1).collect()
    }

    /// Number of bits needed to represent n.
    fn bit_count(n: u64) -> usize {
        if n == 0 {
            1
        } else {
            (64 - n.leading_zeros()) as usize
        }
    }

    /// Compute 2^exp mod m using fast exponentiation.
    fn pow2_mod(exp: usize, m: u64) -> u64 {
        if m == 1 {
            return 0;
        }
        let mut result: u64 = 1;
        let mut base: u64 = 2 % m;
        let mut e = exp as u64;
        while e > 0 {
            if e & 1 == 1 {
                result = result.wrapping_mul(base) % m;
            }
            e >>= 1;
            base = base.wrapping_mul(base) % m;
        }
        result
    }
}

impl SsdFormulation for BinaryLift {
    fn name(&self) -> &str {
        "Binary Indicator Lifting"
    }

    fn dimensionality(&self, n: u64) -> usize {
        Self::bit_count(n)
    }

    fn parallel(&self, n: u64, divisors: &[u64]) -> Vec<u64> {
        let bits = Self::extract_bits(n);
        let num_bits = bits.len();

        // For each divisor d, compute: remainder = sum(bits[i] * (2^i mod d)) mod d
        // This is a matrix-vector product: M[d][i] = 2^i mod d, result = (M * bits) mod d
        // We parallelize across divisors using rayon.
        divisors
            .par_iter()
            .map(|&d| {
                if d <= 1 {
                    return 0;
                }
                // Build the row of M for this divisor and compute the dot product
                let mut remainder: u64 = 0;
                for i in 0..num_bits {
                    if bits[i] == 1 {
                        let weight = Self::pow2_mod(i, d);
                        remainder = (remainder + weight) % d;
                    }
                }
                remainder
            })
            .collect()
    }
}
