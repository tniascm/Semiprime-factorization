//! Bucket sieve: cache-friendly sieve update accumulation.
//!
//! The sieve region is split into "bucket regions" of size 2^LOG_BUCKET_REGION.
//! For each large FB prime, instead of striding through the sieve array (cache-hostile),
//! we push compact 3-byte updates into the target bucket. Then for each bucket region
//! (which fits in L1 cache), we gather all updates and apply them locally.

/// Log2 of bucket region size. 16 = 64KB for x86, 17 = 128KB for Apple Silicon.
pub const LOG_BUCKET_REGION: u32 = 16;
pub const BUCKET_REGION: usize = 1 << LOG_BUCKET_REGION;

/// A compact sieve update: position within bucket region + log contribution.
#[derive(Debug, Clone, Copy)]
#[repr(C, packed)]
pub struct BucketUpdate {
    /// Position within the bucket region (0..BUCKET_REGION-1).
    pub pos: u16,
    /// Quantized log2(p) to subtract from sieve array.
    pub logp: u8,
}

impl BucketUpdate {
    /// Read the position field safely from the packed struct (returns a copy).
    #[inline(always)]
    pub fn position(self) -> u16 {
        self.pos
    }

    /// Read the log-prime field safely from the packed struct (returns a copy).
    #[inline(always)]
    pub fn log_prime(self) -> u8 {
        self.logp
    }
}

/// Bucket array: pre-allocated storage for all bucket updates.
///
/// The flat `data` vector is partitioned into contiguous slices, one per bucket.
/// Each bucket's slice starts at `starts[i]` and extends for `updates_per_bucket`
/// entries. `write_pos[i]` tracks the next write position within bucket `i`'s
/// slice. This design avoids per-bucket heap allocations and keeps updates
/// cache-local during the gather phase.
pub struct BucketArray {
    /// Flat storage for all updates across all buckets.
    data: Vec<BucketUpdate>,
    /// Per-bucket: current write position (index into data).
    write_pos: Vec<usize>,
    /// Per-bucket: start position in data.
    starts: Vec<usize>,
    /// Per-bucket: end position (exclusive) in data — the capacity boundary.
    ends: Vec<usize>,
    /// Number of buckets.
    n_buckets: usize,
}

impl BucketArray {
    /// Allocate bucket array for `n_buckets` buckets with estimated `updates_per_bucket`.
    ///
    /// Each bucket receives a contiguous slice of `updates_per_bucket` entries in the
    /// flat data vector. The total allocation is `n_buckets * updates_per_bucket` entries.
    ///
    /// # Panics
    ///
    /// Panics if `n_buckets` is zero.
    pub fn new(n_buckets: usize, updates_per_bucket: usize) -> Self {
        assert!(n_buckets > 0, "BucketArray: n_buckets must be > 0");

        let total = n_buckets * updates_per_bucket;
        let data = vec![BucketUpdate { pos: 0, logp: 0 }; total];

        let mut starts = Vec::with_capacity(n_buckets);
        let mut ends = Vec::with_capacity(n_buckets);
        let mut write_pos = Vec::with_capacity(n_buckets);

        for i in 0..n_buckets {
            let s = i * updates_per_bucket;
            starts.push(s);
            ends.push(s + updates_per_bucket);
            write_pos.push(s);
        }

        BucketArray {
            data,
            write_pos,
            starts,
            ends,
            n_buckets,
        }
    }

    /// Push an update into the specified bucket.
    ///
    /// # Panics
    ///
    /// Panics if `bucket >= n_buckets` or if the bucket's pre-allocated capacity
    /// is exhausted. In a production NFS implementation the capacity would be
    /// grown dynamically; for now a clear panic message aids tuning.
    #[inline(always)]
    /// Push a bucket update. Hot path — inlined with unchecked indexing in release.
    ///
    /// # Safety invariant
    /// Caller must ensure `bucket < n_buckets` and that the bucket has capacity.
    /// These are checked via `debug_assert` in debug builds.
    #[inline(always)]
    pub fn push(&mut self, bucket: usize, update: BucketUpdate) {
        debug_assert!(bucket < self.n_buckets);
        let wp = unsafe { *self.write_pos.get_unchecked(bucket) };
        debug_assert!(wp < unsafe { *self.ends.get_unchecked(bucket) });
        unsafe {
            *self.data.get_unchecked_mut(wp) = update;
            *self.write_pos.get_unchecked_mut(bucket) = wp + 1;
        }
    }

    /// Get all updates for a bucket region.
    ///
    /// Returns a slice of all updates that have been pushed into `bucket` since
    /// the last `clear()`.
    ///
    /// # Panics
    ///
    /// Panics if `bucket >= n_buckets`.
    pub fn updates_for_bucket(&self, bucket: usize) -> &[BucketUpdate] {
        debug_assert!(
            bucket < self.n_buckets,
            "BucketArray::updates_for_bucket: bucket index {} out of range (n_buckets = {})",
            bucket,
            self.n_buckets,
        );

        &self.data[self.starts[bucket]..self.write_pos[bucket]]
    }

    /// Clear all buckets for reuse with next special-q.
    ///
    /// Resets all write positions to their respective start positions. The
    /// underlying data is not zeroed — only the write cursors are rewound.
    pub fn clear(&mut self) {
        for i in 0..self.n_buckets {
            self.write_pos[i] = self.starts[i];
        }
    }

    /// Number of buckets in this array.
    pub fn n_buckets(&self) -> usize {
        self.n_buckets
    }

    /// Total number of updates across all buckets.
    pub fn total_updates(&self) -> usize {
        self.write_pos
            .iter()
            .zip(self.starts.iter())
            .map(|(wp, s)| wp - s)
            .sum()
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bucket_push_and_read() {
        let mut ba = BucketArray::new(4, 100);
        ba.push(0, BucketUpdate { pos: 10, logp: 5 });
        ba.push(0, BucketUpdate { pos: 20, logp: 3 });
        ba.push(2, BucketUpdate { pos: 15, logp: 7 });

        let b0 = ba.updates_for_bucket(0);
        assert_eq!(b0.len(), 2);
        assert_eq!(b0[0].position(), 10);
        assert_eq!(b0[1].position(), 20);

        let b1 = ba.updates_for_bucket(1);
        assert_eq!(b1.len(), 0);

        let b2 = ba.updates_for_bucket(2);
        assert_eq!(b2.len(), 1);
        assert_eq!(b2[0].position(), 15);

        assert_eq!(ba.total_updates(), 3);
    }

    #[test]
    fn test_bucket_clear() {
        let mut ba = BucketArray::new(4, 100);
        ba.push(0, BucketUpdate { pos: 10, logp: 5 });
        ba.clear();
        assert_eq!(ba.updates_for_bucket(0).len(), 0);
        assert_eq!(ba.total_updates(), 0);
    }

    #[test]
    fn test_bucket_many_updates() {
        let mut ba = BucketArray::new(8, 1000);
        for i in 0..1000u16 {
            ba.push((i % 8) as usize, BucketUpdate { pos: i, logp: 5 });
        }
        assert_eq!(ba.total_updates(), 1000);
        for b in 0..8 {
            assert_eq!(ba.updates_for_bucket(b).len(), 125);
        }
    }

    #[test]
    fn test_bucket_constants() {
        assert_eq!(BUCKET_REGION, 65536);
        assert_eq!(LOG_BUCKET_REGION, 16);
    }

    #[test]
    fn test_bucket_update_size() {
        // BucketUpdate must be exactly 3 bytes (compact for cache efficiency).
        assert_eq!(std::mem::size_of::<BucketUpdate>(), 3);
    }

    #[test]
    fn test_bucket_reuse_after_clear() {
        let mut ba = BucketArray::new(2, 50);
        for i in 0..50u16 {
            ba.push(0, BucketUpdate { pos: i, logp: 1 });
        }
        assert_eq!(ba.total_updates(), 50);

        ba.clear();
        assert_eq!(ba.total_updates(), 0);

        // Re-fill after clear should work identically.
        for i in 0..30u16 {
            ba.push(1, BucketUpdate { pos: i, logp: 2 });
        }
        assert_eq!(ba.total_updates(), 30);
        assert_eq!(ba.updates_for_bucket(0).len(), 0);
        assert_eq!(ba.updates_for_bucket(1).len(), 30);
    }

    #[test]
    fn test_bucket_update_ordering() {
        // Updates should be returned in push order (FIFO within each bucket).
        let mut ba = BucketArray::new(1, 100);
        let positions: Vec<u16> = vec![42, 7, 100, 3, 65535];
        for &p in &positions {
            ba.push(0, BucketUpdate { pos: p, logp: 10 });
        }
        let updates = ba.updates_for_bucket(0);
        for (i, &p) in positions.iter().enumerate() {
            assert_eq!(
                updates[i].position(),
                p,
                "update {} should have pos {}",
                i,
                p
            );
        }
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic]
    fn test_bucket_overflow_panics() {
        let mut ba = BucketArray::new(1, 2);
        ba.push(0, BucketUpdate { pos: 0, logp: 1 });
        ba.push(0, BucketUpdate { pos: 1, logp: 1 });
        // Third push should panic via debug_assert: capacity is 2.
        ba.push(0, BucketUpdate { pos: 2, logp: 1 });
    }
}
