//! Per-bucket-region processing: the inner loop of the NFS sieve.
//!
//! For each bucket region (64KB, fits in L1 cache), this module:
//! 1. Applies bucket updates (large primes) using saturated subtraction
//! 2. Scans for survivors (positions where sieve value is below threshold)

use crate::sieve::bucket::{BucketUpdate, BUCKET_REGION};

/// A sieve survivor: position that passed the threshold test.
#[derive(Debug, Clone, Copy)]
pub struct Survivor {
    /// Position within the bucket region.
    pub pos: u16,
    /// i-coordinate in q-lattice.
    pub i: i64,
    /// j-coordinate in q-lattice.
    pub j: u64,
}

/// Apply bucket updates to sieve array using saturated subtraction.
///
/// This is critical for performance: the inner loop runs billions of times.
/// Each update subtracts `logp` from `sieve[pos]`, clamping at zero.
///
/// # Safety note
///
/// `BucketUpdate` is `#[repr(C, packed)]`. We read fields via the by-value
/// accessor methods (`position()`, `log_prime()`) to avoid creating unaligned
/// references.
#[inline(always)]
pub fn apply_bucket_updates(sieve: &mut [u8], updates: &[BucketUpdate]) {
    for u in updates {
        // Copy out of the packed struct to avoid unaligned access.
        let upd = *u;
        let pos = upd.position() as usize;
        let logp = upd.log_prime();
        if pos < sieve.len() {
            sieve[pos] = sieve[pos].saturating_sub(logp);
        }
    }
}

/// Scan sieve arrays for survivors.
///
/// A survivor is a position where BOTH rational and algebraic sieve values
/// are at or below their respective bounds. Returns positions within the
/// bucket region.
pub fn scan_survivors(
    rat_sieve: &[u8],
    alg_sieve: &[u8],
    rat_bound: u8,
    alg_bound: u8,
) -> Vec<u16> {
    let mut survivors = Vec::new();
    let len = rat_sieve.len().min(alg_sieve.len());
    for i in 0..len {
        if rat_sieve[i] <= rat_bound && alg_sieve[i] <= alg_bound {
            survivors.push(i as u16);
        }
    }
    survivors
}

/// Convert a survivor position in a bucket region to (i, j) q-lattice coordinates.
///
/// The sieve region is organized as: for j in `[0, J)`, for i in `[-I, I)`,
/// the 1D index = `j * (2*I) + (i + I)`.
///
/// Given a bucket region index and position within that region:
/// - `global_pos = bucket_idx * BUCKET_REGION + pos`
/// - `j = global_pos / sieve_width`
/// - `i = (global_pos % sieve_width) - half_i`
pub fn pos_to_ij(
    bucket_idx: usize,
    pos: u16,
    sieve_width: usize, // = 2*I
    half_i: i64,         // = I
) -> (i64, u64) {
    let global_pos = bucket_idx * BUCKET_REGION + pos as usize;
    let j = (global_pos / sieve_width) as u64;
    let i = (global_pos % sieve_width) as i64 - half_i;
    (i, j)
}

/// Process one bucket region: apply updates, scan for survivors.
///
/// Returns survivor positions (as offsets within the bucket region).
/// The caller is responsible for:
///   1. Initializing the norm arrays (calling `init_norm_rat`/`init_norm_alg`)
///   2. Applying the small sieve (calling `small_sieve_region`)
/// before calling this function.
///
/// This function only handles the bucket update application + survivor scan.
pub fn process_bucket_region(
    rat_sieve: &mut [u8],
    alg_sieve: &mut [u8],
    rat_updates: &[BucketUpdate],
    alg_updates: &[BucketUpdate],
    rat_bound: u8,
    alg_bound: u8,
) -> Vec<u16> {
    // Apply bucket updates (large primes)
    apply_bucket_updates(rat_sieve, rat_updates);
    apply_bucket_updates(alg_sieve, alg_updates);

    // Scan for survivors
    scan_survivors(rat_sieve, alg_sieve, rat_bound, alg_bound)
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sieve::bucket::BucketUpdate;

    #[test]
    fn test_apply_bucket_updates() {
        let mut sieve = vec![100u8; 10];
        let updates = vec![
            BucketUpdate { pos: 2, logp: 10 },
            BucketUpdate { pos: 5, logp: 20 },
            BucketUpdate { pos: 2, logp: 15 }, // same position, accumulates
        ];
        apply_bucket_updates(&mut sieve, &updates);
        assert_eq!(sieve[2], 75); // 100 - 10 - 15
        assert_eq!(sieve[5], 80); // 100 - 20
        assert_eq!(sieve[0], 100); // untouched
    }

    #[test]
    fn test_apply_bucket_updates_saturates() {
        let mut sieve = vec![5u8; 10];
        let updates = vec![BucketUpdate { pos: 3, logp: 20 }];
        apply_bucket_updates(&mut sieve, &updates);
        assert_eq!(sieve[3], 0); // saturates at 0, doesn't wrap
    }

    #[test]
    fn test_scan_survivors() {
        let rat = vec![10u8, 5, 100, 3, 50];
        let alg = vec![8u8, 100, 5, 2, 50];
        // Both <= 10:
        // pos 0: rat=10, alg=8 -> both <= 10 -> survivor
        // pos 1: rat=5, alg=100 -> alg > 10 -> no
        // pos 2: rat=100, alg=5 -> rat > 10 -> no
        // pos 3: rat=3, alg=2 -> both <= 10 -> survivor
        // pos 4: rat=50, alg=50 -> both > 10 -> no
        let survivors = scan_survivors(&rat, &alg, 10, 10);
        assert_eq!(survivors, vec![0, 3]);
    }

    #[test]
    fn test_scan_survivors_none() {
        let rat = vec![100u8; 10];
        let alg = vec![100u8; 10];
        let survivors = scan_survivors(&rat, &alg, 10, 10);
        assert!(survivors.is_empty());
    }

    #[test]
    fn test_pos_to_ij() {
        // sieve_width = 1024 (2*512), half_i = 512
        let (i, j) = pos_to_ij(0, 0, 1024, 512);
        assert_eq!(i, -512);
        assert_eq!(j, 0);

        let (i, j) = pos_to_ij(0, 512, 1024, 512);
        assert_eq!(i, 0);
        assert_eq!(j, 0);

        let (i, j) = pos_to_ij(0, 1023, 1024, 512);
        assert_eq!(i, 511);
        assert_eq!(j, 0);

        // Second row
        let (i, j) = pos_to_ij(0, 1024, 1024, 512);
        assert_eq!(i, -512);
        assert_eq!(j, 1);
    }

    #[test]
    fn test_process_bucket_region() {
        let mut rat = vec![50u8; 10];
        let mut alg = vec![50u8; 10];
        let rat_updates = vec![
            BucketUpdate { pos: 3, logp: 45 }, // rat[3] = 50 - 45 = 5
        ];
        let alg_updates = vec![
            BucketUpdate { pos: 3, logp: 48 }, // alg[3] = 50 - 48 = 2
        ];
        let survivors =
            process_bucket_region(&mut rat, &mut alg, &rat_updates, &alg_updates, 10, 10);
        assert_eq!(survivors, vec![3]); // only position 3 has both below threshold
    }
}
