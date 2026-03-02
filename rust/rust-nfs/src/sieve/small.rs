//! Small sieve: line sieve for primes smaller than the bucket threshold.
//!
//! For small primes (below the bucket region size), we use a classical line sieve
//! instead of the bucket approach. For each small prime `p` with root `r`, we stride
//! through the sieve array at step `p`, subtracting `log(p)` at each hit.
//!
//! The key computation is finding the starting position for each prime in each row
//! of the q-lattice.

use crate::arith::mod_inverse;
use crate::sieve::lattice::QLattice;

/// Precomputed small-sieve info for one prime on one side.
///
/// For row `j`, the first hit in the sieve row is at position
/// `(root_i * j) mod p` (with appropriate unsigned conversion),
/// and subsequent hits are spaced `p` apart.
#[derive(Debug, Clone, Copy)]
pub struct SmallSieveEntry {
    /// The factor base prime.
    pub p: u64,
    /// Quantized log2(p) to subtract from sieve cells.
    pub logp: u8,
    /// Transformed root in q-lattice coordinates.
    ///
    /// For row `j`, the first hit position (mod `p`) is `(root_i * j_unsigned) mod p`.
    pub root_i: u64,
}

/// Precompute small sieve entries for the rational side.
///
/// For each factor base prime `p`: the rational polynomial has root `m` (i.e.,
/// `F_rat(a, b) = a - m*b`, so `p | F_rat` when `a === m*b (mod p)`).
///
/// In q-lattice coordinates: `a = a0*i + a1*j`, `b = b0*i + b1*j`, so
/// `(a0 - m*b0)*i === -(a1 - m*b1)*j (mod p)`.
///
/// If `a0 - m*b0 !== 0 (mod p)`, then:
/// `i === -(a1 - m*b1) * (a0 - m*b0)^{-1} * j (mod p)`
///
/// and `root_i = -(a1 - m*b1) * (a0 - m*b0)^{-1} mod p`.
///
/// If the denominator is 0 mod `p`, the prime has a projective root (all positions
/// in the row are hit, or none are). These are skipped (returned with `p = 0` as
/// a sentinel that callers can filter).
pub fn precompute_small_sieve_rat(
    primes: &[u64],
    log_p: &[u8],
    m: u64,
    qlat: &QLattice,
) -> Vec<SmallSieveEntry> {
    assert_eq!(primes.len(), log_p.len());

    let mut entries = Vec::with_capacity(primes.len());

    for (idx, &p) in primes.iter().enumerate() {
        if p == 0 {
            continue;
        }
        let p_i128 = p as i128;

        // Compute (a0 - m*b0) mod p
        let denom = ((qlat.a0 as i128 - (m as i128) * (qlat.b0 as i128)) % p_i128 + p_i128)
            % p_i128;

        if denom == 0 {
            // Projective root: skip this prime for the line sieve.
            // The caller can handle these separately if needed.
            continue;
        }

        let inv = match mod_inverse(denom as u64, p) {
            Some(v) => v,
            None => continue, // gcd(denom, p) > 1 — shouldn't happen for prime p
        };

        // Compute -(a1 - m*b1) mod p
        let numer = ((-(qlat.a1 as i128 - (m as i128) * (qlat.b1 as i128))) % p_i128 + p_i128)
            % p_i128;

        let root_i = ((numer as u128 * inv as u128) % p as u128) as u64;

        entries.push(SmallSieveEntry {
            p,
            logp: log_p[idx],
            root_i,
        });
    }

    entries
}

/// Precompute small sieve entries for the algebraic side.
///
/// For each `(prime, root)` pair: the algebraic polynomial has root `r` mod `p`
/// (i.e., `p | F_alg(a, b)` when `a === r*b (mod p)`).
///
/// In q-lattice coordinates the transformation is identical to the rational case
/// with `m` replaced by `r`:
///
/// `i === -(a1 - r*b1) * (a0 - r*b0)^{-1} * j (mod p)`
///
/// Multiple roots per prime produce multiple entries.
pub fn precompute_small_sieve_alg(
    primes: &[u64],
    roots: &[Vec<u64>],
    log_p: &[u8],
    qlat: &QLattice,
) -> Vec<SmallSieveEntry> {
    assert_eq!(primes.len(), roots.len());
    assert_eq!(primes.len(), log_p.len());

    let mut entries = Vec::new();

    for (idx, &p) in primes.iter().enumerate() {
        if p == 0 {
            continue;
        }
        let p_i128 = p as i128;

        for &r in &roots[idx] {
            let denom = ((qlat.a0 as i128 - (r as i128) * (qlat.b0 as i128)) % p_i128 + p_i128)
                % p_i128;

            if denom == 0 {
                continue;
            }

            let inv = match mod_inverse(denom as u64, p) {
                Some(v) => v,
                None => continue,
            };

            let numer = ((-(qlat.a1 as i128 - (r as i128) * (qlat.b1 as i128))) % p_i128
                + p_i128)
                % p_i128;

            let root_i = ((numer as u128 * inv as u128) % p as u128) as u64;

            entries.push(SmallSieveEntry {
                p,
                logp: log_p[idx],
                root_i,
            });
        }
    }

    entries
}

/// Apply small sieve to a region of the sieve array for one row `j`.
///
/// For each entry, the first hit in the full sieve row is at position
/// `start = (root_i * j_unsigned) mod p`. We then stride at step `p`,
/// subtracting `logp` at each hit using saturating subtraction (no underflow).
///
/// The `region_start` and `region_len` parameters allow sieving a sub-region of the
/// full sieve row (useful when the sieve is processed in bucket-sized chunks).
/// The `sieve` slice corresponds to positions `[region_start, region_start + region_len)`.
pub fn small_sieve_region(
    sieve: &mut [u8],
    entries: &[SmallSieveEntry],
    j: i32,
    region_start: usize,
    region_len: usize,
    _sieve_width: usize,
) {
    let region_end = region_start + region_len;

    for entry in entries {
        let p = entry.p as usize;
        if p == 0 {
            continue;
        }
        let logp = entry.logp;

        // Compute first hit position in the full sieve row.
        // j can be negative; we need (root_i * j) mod p with proper unsigned handling.
        let j_mod_p = ((j as i64).rem_euclid(p as i64)) as u64;
        let start_in_row =
            ((entry.root_i as u128 * j_mod_p as u128) % entry.p as u128) as usize;

        // Find the first hit at or after region_start.
        let first_hit = if start_in_row >= region_start {
            start_in_row
        } else {
            // Advance from start_in_row to the first position >= region_start.
            let gap = region_start - start_in_row;
            let steps = (gap + p - 1) / p; // ceiling division
            start_in_row + steps * p
        };

        // Stride through the region.
        let mut pos = first_hit;
        while pos < region_end {
            let local = pos - region_start;
            sieve[local] = sieve[local].saturating_sub(logp);
            pos += p;
        }
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sieve::lattice::QLattice;

    #[test]
    fn test_small_sieve_identity_lattice() {
        // Identity q-lattice, rational side with m=3
        let qlat = QLattice {
            a0: 1,
            b0: 0,
            a1: 0,
            b1: 1,
        };
        let primes = vec![5u64, 7, 11];
        let log_p = vec![7u8, 8, 10]; // approximate log2 * some scale
        let entries = precompute_small_sieve_rat(&primes, &log_p, 3, &qlat);
        assert_eq!(entries.len(), 3);
        // Each entry should have the original prime
        assert_eq!(entries[0].p, 5);
        assert_eq!(entries[1].p, 7);
        assert_eq!(entries[2].p, 11);
    }

    #[test]
    fn test_small_sieve_identity_lattice_roots() {
        // Identity q-lattice with m=3:
        // denom = a0 - m*b0 = 1 - 0 = 1
        // numer = -(a1 - m*b1) = -(0 - 3) = 3
        // root_i = 3 * inv(1, p) mod p = 3 mod p
        let qlat = QLattice {
            a0: 1,
            b0: 0,
            a1: 0,
            b1: 1,
        };
        let primes = vec![5u64, 7];
        let log_p = vec![7u8, 8];
        let entries = precompute_small_sieve_rat(&primes, &log_p, 3, &qlat);

        // For identity lattice with m=3: root_i = m mod p = 3
        assert_eq!(entries[0].root_i, 3); // 3 mod 5 = 3
        assert_eq!(entries[1].root_i, 3); // 3 mod 7 = 3
    }

    #[test]
    fn test_small_sieve_region_applies_correctly() {
        let entries = vec![SmallSieveEntry {
            p: 5,
            logp: 10,
            root_i: 2,
        }];
        let mut sieve = vec![100u8; 30];
        small_sieve_region(&mut sieve, &entries, 1, 0, 30, 30);
        // For j=1: start = (2 * 1) % 5 = 2
        // Hits at positions 2, 7, 12, 17, 22, 27
        assert_eq!(sieve[2], 90);
        assert_eq!(sieve[7], 90);
        assert_eq!(sieve[12], 90);
        assert_eq!(sieve[17], 90);
        assert_eq!(sieve[22], 90);
        assert_eq!(sieve[27], 90);
        // Non-hit positions unchanged
        assert_eq!(sieve[0], 100);
        assert_eq!(sieve[3], 100);
    }

    #[test]
    fn test_small_sieve_alg_multiple_roots() {
        let qlat = QLattice {
            a0: 1,
            b0: 0,
            a1: 0,
            b1: 1,
        };
        let primes = vec![7u64];
        let roots = vec![vec![2, 5]]; // two roots mod 7
        let log_p = vec![8u8];
        let entries = precompute_small_sieve_alg(&primes, &roots, &log_p, &qlat);
        // Should have 2 entries (one per root)
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].p, 7);
        assert_eq!(entries[1].p, 7);
    }

    #[test]
    fn test_small_sieve_region_sub_region() {
        // Test sieving a sub-region of the full row
        let entries = vec![SmallSieveEntry {
            p: 5,
            logp: 10,
            root_i: 0,
        }];
        let mut sieve = vec![100u8; 10];
        // Region starts at position 10, length 10
        small_sieve_region(&mut sieve, &entries, 5, 10, 10, 100);
        // For j=5: start = (0 * 5) % 5 = 0
        // Hits at global positions 0, 5, 10, 15, ...
        // In region [10, 20): hits at 10 (local 0) and 15 (local 5)
        assert_eq!(sieve[0], 90); // global pos 10
        assert_eq!(sieve[5], 90); // global pos 15
        // Others unchanged
        assert_eq!(sieve[1], 100);
        assert_eq!(sieve[3], 100);
    }

    #[test]
    fn test_small_sieve_region_saturating_sub() {
        // Verify saturating subtraction: don't underflow past 0
        let entries = vec![SmallSieveEntry {
            p: 3,
            logp: 200,
            root_i: 0,
        }];
        let mut sieve = vec![50u8; 10];
        small_sieve_region(&mut sieve, &entries, 3, 0, 10, 10);
        // For j=3: start = (0 * 3) % 3 = 0, hits at 0, 3, 6, 9
        assert_eq!(sieve[0], 0); // 50 - 200 saturates to 0
        assert_eq!(sieve[3], 0);
        assert_eq!(sieve[6], 0);
        assert_eq!(sieve[9], 0);
        assert_eq!(sieve[1], 50); // untouched
    }

    #[test]
    fn test_small_sieve_region_multiple_entries() {
        // Two primes hitting the same region
        let entries = vec![
            SmallSieveEntry {
                p: 3,
                logp: 10,
                root_i: 0,
            },
            SmallSieveEntry {
                p: 5,
                logp: 15,
                root_i: 0,
            },
        ];
        let mut sieve = vec![100u8; 30];
        small_sieve_region(&mut sieve, &entries, 3, 0, 30, 30);
        // p=3 hits 0,3,6,9,12,15,18,21,24,27
        // p=5 hits 0,5,10,15,20,25
        // Position 0: hit by both -> 100 - 10 - 15 = 75
        assert_eq!(sieve[0], 75);
        // Position 15: hit by both -> 100 - 10 - 15 = 75
        assert_eq!(sieve[15], 75);
        // Position 3: only p=3 -> 90
        assert_eq!(sieve[3], 90);
        // Position 5: only p=5 -> 85
        assert_eq!(sieve[5], 85);
        // Position 1: neither -> 100
        assert_eq!(sieve[1], 100);
    }

    #[test]
    fn test_small_sieve_negative_j() {
        // Test with negative j value
        let entries = vec![SmallSieveEntry {
            p: 7,
            logp: 10,
            root_i: 3,
        }];
        let mut sieve = vec![100u8; 21];
        small_sieve_region(&mut sieve, &entries, -2, 0, 21, 21);
        // j = -2: j_mod_p = (-2).rem_euclid(7) = 5
        // start = (3 * 5) % 7 = 15 % 7 = 1
        // Hits at 1, 8, 15
        assert_eq!(sieve[1], 90);
        assert_eq!(sieve[8], 90);
        assert_eq!(sieve[15], 90);
        assert_eq!(sieve[0], 100);
    }

    #[test]
    fn test_precompute_small_sieve_rat_projective() {
        // When a0 - m*b0 === 0 (mod p), the root is projective and should be skipped.
        // a0=5, b0=1, m=5: denom = 5 - 5*1 = 0 (mod 5)
        let qlat = QLattice {
            a0: 5,
            b0: 1,
            a1: 0,
            b1: 1,
        };
        let primes = vec![5u64];
        let log_p = vec![7u8];
        let entries = precompute_small_sieve_rat(&primes, &log_p, 5, &qlat);
        // p=5 should be skipped (projective root)
        assert_eq!(entries.len(), 0);
    }

    #[test]
    fn test_precompute_small_sieve_alg_identity() {
        // Identity lattice with root r=2, prime p=7
        // denom = a0 - r*b0 = 1 - 0 = 1
        // numer = -(a1 - r*b1) = -(0 - 2) = 2
        // root_i = 2 * inv(1, 7) mod 7 = 2
        let qlat = QLattice {
            a0: 1,
            b0: 0,
            a1: 0,
            b1: 1,
        };
        let primes = vec![7u64];
        let roots = vec![vec![2u64]];
        let log_p = vec![8u8];
        let entries = precompute_small_sieve_alg(&primes, &roots, &log_p, &qlat);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].root_i, 2);
        assert_eq!(entries[0].p, 7);
        assert_eq!(entries[0].logp, 8);
    }
}
