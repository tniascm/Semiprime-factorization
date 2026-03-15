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
    /// For projective roots in q-lattice coordinates, every cell in each
    /// `projective_row_period`-th row is hit. Zero means an affine root.
    pub projective_row_period: u64,
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
/// If the denominator is 0 mod `p`, the transformed root is projective in
/// `(i,j)` coordinates. For prime `p`, that means either:
/// - every cell in rows `j ≡ 0 (mod p)` is hit, or
/// - every cell in every row is hit (the degenerate `numer == 0` case).
///
/// We keep those entries so the small sieve does not silently drop them.
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
        let denom =
            ((qlat.a0 as i128 - (m as i128) * (qlat.b0 as i128)) % p_i128 + p_i128) % p_i128;

        let numer =
            ((-(qlat.a1 as i128 - (m as i128) * (qlat.b1 as i128))) % p_i128 + p_i128) % p_i128;

        if denom == 0 {
            let projective_row_period = if numer == 0 { 1 } else { p };
            entries.push(SmallSieveEntry {
                p,
                logp: log_p[idx],
                projective_row_period,
                root_i: 0,
            });
            continue;
        }

        let inv = match mod_inverse(denom as u64, p) {
            Some(v) => v,
            None => continue, // gcd(denom, p) > 1 — shouldn't happen for prime p
        };

        let root_i = ((numer as u128 * inv as u128) % p as u128) as u64;

        entries.push(SmallSieveEntry {
            p,
            logp: log_p[idx],
            projective_row_period: 0,
            root_i,
        });
    }

    entries
}

/// Precompute small sieve entries for the rational side with general g(x) = g1*x + g0.
///
/// The homogeneous rational polynomial is `G(a,b) = g1*a + g0*b`.
/// In q-lattice coordinates: `a = a0*i + a1*j`, `b = b0*i + b1*j`, so
/// `G(i,j) = (g1*a0 + g0*b0)*i + (g1*a1 + g0*b1)*j`.
///
/// Hit condition: `p | G(i,j)`, giving
/// `i === -(g1*a1 + g0*b1) * (g1*a0 + g0*b0)^{-1} * j (mod p)`.
///
/// When `g1=1, g0=-m`, this reduces to the original `precompute_small_sieve_rat`.
pub fn precompute_small_sieve_rat_g(
    primes: &[u64],
    log_p: &[u8],
    g0: i64,
    g1: i64,
    qlat: &QLattice,
) -> Vec<SmallSieveEntry> {
    assert_eq!(primes.len(), log_p.len());

    let mut entries = Vec::with_capacity(primes.len());

    for (idx, &p) in primes.iter().enumerate() {
        if p == 0 {
            continue;
        }
        let p_i128 = p as i128;

        // denom = (g1*a0 + g0*b0) mod p
        let denom =
            (((g1 as i128) * (qlat.a0 as i128) + (g0 as i128) * (qlat.b0 as i128)) % p_i128 + p_i128) % p_i128;

        // numer = -(g1*a1 + g0*b1) mod p
        let numer =
            ((-((g1 as i128) * (qlat.a1 as i128) + (g0 as i128) * (qlat.b1 as i128))) % p_i128 + p_i128) % p_i128;

        if denom == 0 {
            let projective_row_period = if numer == 0 { 1 } else { p };
            entries.push(SmallSieveEntry {
                p,
                logp: log_p[idx],
                projective_row_period,
                root_i: 0,
            });
            continue;
        }

        let inv = match mod_inverse(denom as u64, p) {
            Some(v) => v,
            None => continue,
        };

        let root_i = ((numer as u128 * inv as u128) % p as u128) as u64;

        entries.push(SmallSieveEntry {
            p,
            logp: log_p[idx],
            projective_row_period: 0,
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
            let denom =
                ((qlat.a0 as i128 - (r as i128) * (qlat.b0 as i128)) % p_i128 + p_i128) % p_i128;

            let numer =
                ((-(qlat.a1 as i128 - (r as i128) * (qlat.b1 as i128))) % p_i128 + p_i128) % p_i128;

            if denom == 0 {
                let projective_row_period = if numer == 0 { 1 } else { p };
                entries.push(SmallSieveEntry {
                    p,
                    logp: log_p[idx],
                    projective_row_period,
                    root_i: 0,
                });
                continue;
            }

            let inv = match mod_inverse(denom as u64, p) {
                Some(v) => v,
                None => continue,
            };

            let root_i = ((numer as u128 * inv as u128) % p as u128) as u64;

            entries.push(SmallSieveEntry {
                p,
                logp: log_p[idx],
                projective_row_period: 0,
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
/// On aarch64, tiny primes (p ≤ 7) use NEON SIMD (16 bytes/op) for ~8-16x speedup
/// since these primes produce the most hits per region.
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
    sieve_width: usize,
) {
    let region_end = region_start + region_len;
    let half_i = sieve_width / 2;

    for entry in entries {
        let logp = entry.logp;
        let row_period = entry.projective_row_period;
        if row_period != 0 {
            if j.rem_euclid(row_period as i32) == 0 {
                for cell in sieve.iter_mut() {
                    *cell = cell.saturating_sub(logp);
                }
            }
            continue;
        }

        let p = entry.p as usize;

        let j_mod_p = ((j as i64).rem_euclid(p as i64)) as u64;
        let start_in_row = ((half_i as u64 + entry.root_i * j_mod_p)
            % entry.p) as usize;

        // Find the first hit at or after region_start.
        let first_hit = if start_in_row >= region_start {
            start_in_row
        } else {
            let gap = region_start - start_in_row;
            let steps = (gap + p - 1) / p;
            start_in_row + steps * p
        };

        let local_start = first_hit.saturating_sub(region_start);

        // Use SIMD for tiny primes (p <= 7) on aarch64.
        #[cfg(target_arch = "aarch64")]
        if p <= 7 {
            small_sieve_simd(sieve, p, logp, local_start, region_len);
            continue;
        }

        // Stride through the region using unchecked access (hot path).
        let mut pos = first_hit;
        while pos < region_end {
            let local = pos - region_start;
            debug_assert!(local < sieve.len());
            unsafe {
                let cell = sieve.get_unchecked_mut(local);
                *cell = cell.saturating_sub(logp);
            }
            pos += p;
        }
    }
}

/// Apply small sieve to a region with incremental position tracking.
///
/// This is an optimized variant of `small_sieve_region` that avoids the
/// per-entry modular division (`(half_i + root_i * j_mod_p) % p`) on
/// consecutive rows. Instead, it maintains a `tracked_starts` array with one
/// element per entry, advancing each position with a single addition and
/// conditional subtract: `pos = (pos + root_i) % p`.
///
/// When `prev_j` is `Some(j - 1)`, positions are advanced incrementally.
/// Otherwise (first call or non-consecutive rows), positions are computed
/// from scratch.
///
/// SIMD entries (p <= 7 on aarch64) and projective entries always compute
/// from scratch, since SIMD already dominates the cost for tiny primes and
/// projective entries have special row logic.
pub fn small_sieve_region_tracked(
    sieve: &mut [u8],
    entries: &[SmallSieveEntry],
    j: i32,
    region_start: usize,
    region_len: usize,
    sieve_width: usize,
    tracked_starts: &mut [usize],
    prev_j: Option<i32>,
) {
    let region_end = region_start + region_len;
    let half_i = sieve_width / 2;
    // Determine the update mode:
    // - Consecutive (prev_j == j-1): advance positions incrementally
    // - Same row (prev_j == j): positions already correct, reuse as-is
    // - Otherwise: recompute from scratch
    let update_mode = match prev_j {
        Some(pj) if pj == j - 1 => 1u8,  // consecutive: advance
        Some(pj) if pj == j => 2u8,       // same row: reuse
        _ => 0u8,                          // recompute from scratch
    };

    for (idx, entry) in entries.iter().enumerate() {
        let logp = entry.logp;
        let row_period = entry.projective_row_period;
        if row_period != 0 {
            // Projective entries: no position tracking, use existing logic.
            if j.rem_euclid(row_period as i32) == 0 {
                for cell in sieve.iter_mut() {
                    *cell = cell.saturating_sub(logp);
                }
            }
            continue;
        }

        let p = entry.p as usize;

        // Determine start_in_row: incrementally, reuse, or from scratch.
        let start_in_row;

        #[cfg(target_arch = "aarch64")]
        let is_simd_prime = p <= 7;
        #[cfg(not(target_arch = "aarch64"))]
        let is_simd_prime = false;

        if is_simd_prime {
            // SIMD primes: always compute from scratch (SIMD already fast).
            let j_mod_p = ((j as i64).rem_euclid(p as i64)) as u64;
            start_in_row = ((half_i as u64 + entry.root_i * j_mod_p) % entry.p) as usize;
            tracked_starts[idx] = start_in_row;
        } else if update_mode == 1 {
            // Consecutive row: advance pos_new = (pos_old + root_i) % p
            let old_pos = tracked_starts[idx];
            let root = entry.root_i as usize;
            let new_pos = old_pos + root;
            start_in_row = if new_pos >= p { new_pos - p } else { new_pos };
            tracked_starts[idx] = start_in_row;
        } else if update_mode == 2 {
            // Same row repeated (split across bucket regions): reuse stored position.
            start_in_row = tracked_starts[idx];
        } else {
            // Non-consecutive or first call: compute from scratch.
            let j_mod_p = ((j as i64).rem_euclid(p as i64)) as u64;
            start_in_row = ((half_i as u64 + entry.root_i * j_mod_p) % entry.p) as usize;
            tracked_starts[idx] = start_in_row;
        }

        // Find the first hit at or after region_start.
        let first_hit = if start_in_row >= region_start {
            start_in_row
        } else {
            let gap = region_start - start_in_row;
            let steps = (gap + p - 1) / p;
            start_in_row + steps * p
        };

        let local_start = first_hit.saturating_sub(region_start);

        // Use SIMD for tiny primes (p <= 7) on aarch64.
        #[cfg(target_arch = "aarch64")]
        if p <= 7 {
            small_sieve_simd(sieve, p, logp, local_start, region_len);
            continue;
        }

        // Stride through the region using unchecked access (hot path).
        let mut pos = first_hit;
        while pos < region_end {
            let local = pos - region_start;
            debug_assert!(local < sieve.len());
            unsafe {
                let cell = sieve.get_unchecked_mut(local);
                *cell = cell.saturating_sub(logp);
            }
            pos += p;
        }
    }
}

/// SIMD pattern sieve for tiny primes (p=2,3,5,7) on aarch64.
///
/// Pre-computes p different 16-byte patterns (one for each possible alignment
/// of the stride-p hits within a 16-byte chunk), then sweeps through the sieve
/// with NEON vqsubq_u8 (saturating byte-wise subtraction, 16 bytes at a time).
///
/// Only processes from `local_start` onward — bytes before the first hit are
/// not touched.
#[cfg(target_arch = "aarch64")]
fn small_sieve_simd(sieve: &mut [u8], p: usize, logp: u8, local_start: usize, region_len: usize) {
    use std::arch::aarch64::*;

    if local_start >= region_len {
        return;
    }

    // Build p patterns, one for each possible offset within a 16-byte chunk.
    // Pattern with offset `off` has logp at positions k where k % p == off.
    let mut patterns_raw = [[0u8; 16]; 8];
    for offset in 0..p {
        for k in 0..16 {
            if k % p == offset {
                patterns_raw[offset][k] = logp;
            }
        }
    }

    unsafe {
        let mut patterns = [vdupq_n_u8(0); 8];
        for i in 0..p {
            patterns[i] = vld1q_u8(patterns_raw[i].as_ptr());
        }

        let ptr = sieve.as_mut_ptr();
        // As we advance 16 bytes, the hit alignment shifts backward by 16 mod p.
        // The offset increases by (p - 16%p) % p to compensate.
        let offset_step = (p - (16 % p)) % p;

        // Handle the first partial chunk (before the first aligned boundary
        // after local_start) with scalar to avoid subtracting before first_hit.
        let first_full_chunk = (local_start + 15) / 16; // ceiling division
        let simd_start = first_full_chunk * 16;

        // Scalar for positions local_start..min(simd_start, region_len)
        let scalar_end = simd_start.min(region_len);
        let mut pos = local_start;
        while pos < scalar_end {
            let cell = sieve.get_unchecked_mut(pos);
            *cell = cell.saturating_sub(logp);
            pos += p;
        }

        if simd_start >= region_len {
            return;
        }

        // For the SIMD region, hits are at local_start, local_start+p, local_start+2p, ...
        // In chunk c (starting at byte c*16), a hit falls at position (local_start % p)
        // relative to the chunk's alignment with stride p.
        // Offset for chunk at simd_start: the first hit at or after simd_start is at
        // some position h. Within that chunk, h - simd_start gives the byte offset.
        // The pattern offset = h % p, but we need it relative to the chunk boundary.
        // Since the pattern has logp where k % p == offset, the right offset for
        // a chunk starting at byte `base` is: (local_start - base) % p, adjusted
        // to be in [0, p).
        let mut current_offset = if simd_start >= local_start {
            // (local_start - simd_start) mod p, but we need the positive representative
            (p - ((simd_start - local_start) % p)) % p
        } else {
            local_start % p
        };

        let n_chunks = (region_len - simd_start) / 16;

        for c in 0..n_chunks {
            let chunk_ptr = ptr.add(simd_start + c * 16);
            let val = vld1q_u8(chunk_ptr);
            let pat = *patterns.get_unchecked(current_offset);
            let result = vqsubq_u8(val, pat);
            vst1q_u8(chunk_ptr, result);
            current_offset += offset_step;
            if current_offset >= p {
                current_offset -= p;
            }
        }

        // Handle remaining bytes after the last full SIMD chunk.
        let remainder_start = simd_start + n_chunks * 16;
        // Find first hit in remainder.
        let mut pos = if local_start >= remainder_start {
            local_start
        } else {
            let gap = remainder_start - local_start;
            let steps = (gap + p - 1) / p;
            local_start + steps * p
        };
        while pos < region_len {
            let cell = sieve.get_unchecked_mut(pos);
            *cell = cell.saturating_sub(logp);
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
            projective_row_period: 0,
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
            projective_row_period: 0,
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
            projective_row_period: 0,
            root_i: 0,
        }];
        let mut sieve = vec![50u8; 10];
        small_sieve_region(&mut sieve, &entries, 3, 0, 10, 10);
        // half_i = 5, so start = (5 + 0*3) % 3 = 2, hits at 2, 5, 8
        assert_eq!(sieve[2], 0); // 50 - 200 saturates to 0
        assert_eq!(sieve[5], 0);
        assert_eq!(sieve[8], 0);
        assert_eq!(sieve[1], 50); // untouched
    }

    #[test]
    fn test_small_sieve_region_multiple_entries() {
        // Two primes hitting the same region
        let entries = vec![
            SmallSieveEntry {
                p: 3,
                logp: 10,
                projective_row_period: 0,
                root_i: 0,
            },
            SmallSieveEntry {
                p: 5,
                logp: 15,
                projective_row_period: 0,
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
            projective_row_period: 0,
            root_i: 3,
        }];
        let mut sieve = vec![100u8; 21];
        small_sieve_region(&mut sieve, &entries, -2, 0, 21, 21);
        // j = -2: j_mod_p = (-2).rem_euclid(7) = 5
        // half_i = 10, start = (10 + 3 * 5) % 7 = 4
        // Hits at 4, 11, 18
        assert_eq!(sieve[4], 90);
        assert_eq!(sieve[11], 90);
        assert_eq!(sieve[18], 90);
        assert_eq!(sieve[0], 100);
    }

    #[test]
    fn test_precompute_small_sieve_rat_projective() {
        // When a0 - m*b0 === 0 (mod p), the transformed root is projective.
        // Here numer != 0, so hits occur on every p-th row.
        let qlat = QLattice {
            a0: 5,
            b0: 1,
            a1: 1,
            b1: 0,
        };
        let primes = vec![5u64];
        let log_p = vec![7u8];
        let entries = precompute_small_sieve_rat(&primes, &log_p, 5, &qlat);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].p, 5);
        assert_eq!(entries[0].projective_row_period, 5);
    }

    #[test]
    fn test_precompute_small_sieve_rat_projective_all_rows() {
        // Degenerate projective case with numer == 0: every row is hit.
        let qlat = QLattice {
            a0: 5,
            b0: 1,
            a1: 0,
            b1: 1,
        };
        let entries = precompute_small_sieve_rat(&[5u64], &[7u8], 5, &qlat);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].projective_row_period, 1);
    }

    #[test]
    fn test_small_sieve_region_projective_hits_full_row() {
        let entries = vec![SmallSieveEntry {
            p: 5,
            logp: 10,
            projective_row_period: 5,
            root_i: 0,
        }];

        let mut sieve = vec![100u8; 8];
        small_sieve_region(&mut sieve, &entries, 10, 0, 8, 8);
        assert!(sieve.iter().all(|&v| v == 90));

        let mut other_row = vec![100u8; 8];
        small_sieve_region(&mut other_row, &entries, 11, 0, 8, 8);
        assert!(other_row.iter().all(|&v| v == 100));
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

    // ---------------------------------------------------------------
    // Tests for small_sieve_region_tracked
    // ---------------------------------------------------------------

    /// Helper: run both tracked and untracked on the same inputs and assert
    /// identical sieve output.
    fn assert_tracked_matches_untracked(
        entries: &[SmallSieveEntry],
        j: i32,
        region_start: usize,
        region_len: usize,
        sieve_width: usize,
        tracked_starts: &mut [usize],
        prev_j: Option<i32>,
    ) {
        let mut sieve_ref = vec![100u8; region_len];
        let mut sieve_trk = vec![100u8; region_len];

        small_sieve_region(
            &mut sieve_ref,
            entries,
            j,
            region_start,
            region_len,
            sieve_width,
        );
        small_sieve_region_tracked(
            &mut sieve_trk,
            entries,
            j,
            region_start,
            region_len,
            sieve_width,
            tracked_starts,
            prev_j,
        );

        assert_eq!(
            sieve_ref, sieve_trk,
            "tracked diverged from reference at j={}, region_start={}, prev_j={:?}",
            j, region_start, prev_j
        );
    }

    #[test]
    fn test_tracked_matches_untracked_from_scratch() {
        let entries = vec![
            SmallSieveEntry { p: 11, logp: 10, projective_row_period: 0, root_i: 3 },
            SmallSieveEntry { p: 13, logp: 11, projective_row_period: 0, root_i: 7 },
            SmallSieveEntry { p: 17, logp: 12, projective_row_period: 0, root_i: 5 },
        ];
        let mut tracked = vec![0usize; entries.len()];
        // From scratch (prev_j = None)
        for j in 0..20 {
            assert_tracked_matches_untracked(&entries, j, 0, 60, 60, &mut tracked, None);
        }
    }

    #[test]
    fn test_tracked_consecutive_rows() {
        let entries = vec![
            SmallSieveEntry { p: 11, logp: 10, projective_row_period: 0, root_i: 3 },
            SmallSieveEntry { p: 13, logp: 11, projective_row_period: 0, root_i: 7 },
            SmallSieveEntry { p: 59, logp: 16, projective_row_period: 0, root_i: 22 },
        ];
        let mut tracked = vec![0usize; entries.len()];
        let sieve_width = 80;

        // First row from scratch
        assert_tracked_matches_untracked(&entries, 0, 0, sieve_width, sieve_width, &mut tracked, None);

        // Subsequent rows incrementally
        for j in 1..50 {
            assert_tracked_matches_untracked(
                &entries, j, 0, sieve_width, sieve_width, &mut tracked, Some(j - 1),
            );
        }
    }

    #[test]
    fn test_tracked_negative_j_consecutive() {
        let entries = vec![
            SmallSieveEntry { p: 7, logp: 8, projective_row_period: 0, root_i: 3 },
            SmallSieveEntry { p: 23, logp: 13, projective_row_period: 0, root_i: 11 },
        ];
        let mut tracked = vec![0usize; entries.len()];
        let sieve_width = 40;

        // Start with a negative j
        assert_tracked_matches_untracked(&entries, -5, 0, sieve_width, sieve_width, &mut tracked, None);

        for j in -4..10 {
            assert_tracked_matches_untracked(
                &entries, j, 0, sieve_width, sieve_width, &mut tracked, Some(j - 1),
            );
        }
    }

    #[test]
    fn test_tracked_same_row_split_region() {
        // Simulate a row split across two bucket regions
        let entries = vec![
            SmallSieveEntry { p: 11, logp: 10, projective_row_period: 0, root_i: 4 },
        ];
        let mut tracked = vec![0usize; entries.len()];
        let sieve_width = 80;

        // First half of row j=3 (region_start=0, region_len=40)
        assert_tracked_matches_untracked(&entries, 3, 0, 40, sieve_width, &mut tracked, None);

        // Second half of same row j=3 (region_start=40, region_len=40), prev_j=3
        assert_tracked_matches_untracked(&entries, 3, 40, 40, sieve_width, &mut tracked, Some(3));
    }

    #[test]
    fn test_tracked_non_consecutive_recompute() {
        let entries = vec![
            SmallSieveEntry { p: 17, logp: 12, projective_row_period: 0, root_i: 9 },
        ];
        let mut tracked = vec![0usize; entries.len()];
        let sieve_width = 60;

        // Process row 5
        assert_tracked_matches_untracked(&entries, 5, 0, sieve_width, sieve_width, &mut tracked, None);

        // Jump to row 10 (non-consecutive)
        assert_tracked_matches_untracked(&entries, 10, 0, sieve_width, sieve_width, &mut tracked, Some(5));
    }

    #[test]
    fn test_tracked_projective_entry() {
        let entries = vec![
            SmallSieveEntry { p: 5, logp: 10, projective_row_period: 5, root_i: 0 },
            SmallSieveEntry { p: 11, logp: 10, projective_row_period: 0, root_i: 3 },
        ];
        let mut tracked = vec![0usize; entries.len()];
        let sieve_width = 30;

        // From scratch
        assert_tracked_matches_untracked(&entries, 0, 0, sieve_width, sieve_width, &mut tracked, None);

        // Consecutive rows (projective should still work correctly)
        for j in 1..15 {
            assert_tracked_matches_untracked(
                &entries, j, 0, sieve_width, sieve_width, &mut tracked, Some(j - 1),
            );
        }
    }

    #[test]
    fn test_tracked_sub_region() {
        // Test with region_start > 0 (sub-region of full row)
        let entries = vec![
            SmallSieveEntry { p: 11, logp: 10, projective_row_period: 0, root_i: 4 },
            SmallSieveEntry { p: 19, logp: 12, projective_row_period: 0, root_i: 7 },
        ];
        let mut tracked = vec![0usize; entries.len()];
        let sieve_width = 100;

        // Region starts at offset 30
        assert_tracked_matches_untracked(&entries, 0, 30, 40, sieve_width, &mut tracked, None);

        // Consecutive row, same sub-region offset
        assert_tracked_matches_untracked(&entries, 1, 30, 40, sieve_width, &mut tracked, Some(0));

        // Consecutive row, different sub-region offset
        assert_tracked_matches_untracked(&entries, 2, 50, 30, sieve_width, &mut tracked, Some(1));
    }

    #[test]
    fn test_tracked_many_entries_full_sweep() {
        // Many entries, sweep through consecutive rows to stress test
        let entries: Vec<SmallSieveEntry> = (0..20)
            .map(|i| {
                let p = [11, 13, 17, 19, 23, 29, 31, 37, 41, 43,
                          47, 53, 59, 61, 67, 71, 73, 79, 83, 89][i];
                SmallSieveEntry {
                    p,
                    logp: (p as f64).log2().round() as u8,
                    projective_row_period: 0,
                    root_i: (p / 3) as u64,
                }
            })
            .collect();
        let mut tracked = vec![0usize; entries.len()];
        let sieve_width = 200;

        // First row from scratch
        assert_tracked_matches_untracked(&entries, 0, 0, sieve_width, sieve_width, &mut tracked, None);

        // 100 consecutive rows
        for j in 1..100 {
            assert_tracked_matches_untracked(
                &entries, j, 0, sieve_width, sieve_width, &mut tracked, Some(j - 1),
            );
        }
    }
}
