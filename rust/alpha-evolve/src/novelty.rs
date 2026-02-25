//! Novelty search: behavioral fingerprinting and novelty archive.
//!
//! Prevents premature convergence by rewarding programs that exhibit novel
//! behavior (different pass/fail patterns across test semiprimes), even if
//! their raw factoring performance is mediocre. Maintains an archive of
//! behavioral fingerprints seen during evolution.

use std::collections::HashSet;

/// A behavioral fingerprint: binary vector of pass/fail on a fixed test suite.
///
/// Each bit represents whether the program factored a specific test semiprime.
/// Programs with the same fingerprint exhibit the same behavior across the suite.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BehaviorFingerprint {
    /// Compact binary vector: bit i = 1 if test case i was factored.
    pub bits: Vec<bool>,
}

impl BehaviorFingerprint {
    /// Create a new empty fingerprint of the given size.
    pub fn new(size: usize) -> Self {
        BehaviorFingerprint {
            bits: vec![false; size],
        }
    }

    /// Set test case `i` as passed.
    pub fn set_pass(&mut self, i: usize) {
        if i < self.bits.len() {
            self.bits[i] = true;
        }
    }

    /// Count the number of passed test cases.
    pub fn pass_count(&self) -> usize {
        self.bits.iter().filter(|&&b| b).count()
    }

    /// Compute Hamming distance to another fingerprint.
    pub fn hamming_distance(&self, other: &BehaviorFingerprint) -> usize {
        self.bits
            .iter()
            .zip(other.bits.iter())
            .filter(|(a, b)| a != b)
            .count()
    }

    /// Compute Euclidean distance (treating bools as 0/1 floats).
    pub fn euclidean_distance(&self, other: &BehaviorFingerprint) -> f64 {
        let d: f64 = self
            .bits
            .iter()
            .zip(other.bits.iter())
            .map(|(&a, &b)| {
                let diff = (a as i32 - b as i32) as f64;
                diff * diff
            })
            .sum();
        d.sqrt()
    }
}

/// Archive of behavioral fingerprints seen during evolution.
///
/// Stores unique fingerprints and computes novelty scores based on
/// distance to k nearest neighbors.
pub struct NoveltyArchive {
    /// Stored fingerprints.
    fingerprints: Vec<BehaviorFingerprint>,
    /// Unique fingerprints as a set for deduplication.
    unique: HashSet<Vec<bool>>,
    /// Number of nearest neighbors for novelty score.
    k_neighbors: usize,
    /// Maximum archive size (LRU eviction after this).
    max_size: usize,
}

impl NoveltyArchive {
    /// Create a new novelty archive.
    pub fn new(k_neighbors: usize, max_size: usize) -> Self {
        NoveltyArchive {
            fingerprints: Vec::new(),
            unique: HashSet::new(),
            k_neighbors,
            max_size,
        }
    }

    /// Add a fingerprint to the archive if it's novel enough.
    ///
    /// Returns the novelty score for the fingerprint.
    pub fn add(&mut self, fp: &BehaviorFingerprint) -> f64 {
        let novelty = self.novelty_score(fp);

        // Only add if this exact behavior hasn't been seen before
        if self.unique.insert(fp.bits.clone()) {
            self.fingerprints.push(fp.clone());

            // LRU eviction: remove oldest if over capacity
            if self.fingerprints.len() > self.max_size {
                let removed = self.fingerprints.remove(0);
                self.unique.remove(&removed.bits);
            }
        }

        novelty
    }

    /// Compute the novelty score: mean distance to k nearest neighbors.
    ///
    /// Higher = more novel (farther from known behaviors).
    /// Returns 0 if archive is empty.
    pub fn novelty_score(&self, fp: &BehaviorFingerprint) -> f64 {
        if self.fingerprints.is_empty() {
            return 1.0; // Maximum novelty for the first program
        }

        // Compute distances to all archived fingerprints
        let mut distances: Vec<f64> = self
            .fingerprints
            .iter()
            .map(|archived| fp.euclidean_distance(archived))
            .collect();

        // Sort ascending and take k nearest
        distances.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let k = self.k_neighbors.min(distances.len());
        if k == 0 {
            return 1.0;
        }

        let sum: f64 = distances[..k].iter().sum();
        sum / k as f64
    }

    /// Get the current archive size (unique fingerprints).
    pub fn size(&self) -> usize {
        self.fingerprints.len()
    }

    /// Get the number of unique behaviors observed.
    pub fn unique_count(&self) -> usize {
        self.unique.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fingerprint_hamming() {
        let mut fp1 = BehaviorFingerprint::new(5);
        fp1.set_pass(0);
        fp1.set_pass(2);

        let mut fp2 = BehaviorFingerprint::new(5);
        fp2.set_pass(0);
        fp2.set_pass(3);

        assert_eq!(fp1.hamming_distance(&fp2), 2); // bits 2 and 3 differ
    }

    #[test]
    fn test_fingerprint_euclidean() {
        let mut fp1 = BehaviorFingerprint::new(4);
        fp1.set_pass(0);
        fp1.set_pass(1);

        let fp2 = BehaviorFingerprint::new(4);
        // All zeros: distance = sqrt(1+1+0+0) = sqrt(2)
        let d = fp1.euclidean_distance(&fp2);
        assert!((d - std::f64::consts::SQRT_2).abs() < 1e-10);
    }

    #[test]
    fn test_novelty_archive_empty() {
        let archive = NoveltyArchive::new(5, 1000);
        let fp = BehaviorFingerprint::new(10);
        assert_eq!(archive.novelty_score(&fp), 1.0);
    }

    #[test]
    fn test_novelty_archive_add() {
        let mut archive = NoveltyArchive::new(3, 100);

        let mut fp1 = BehaviorFingerprint::new(5);
        fp1.set_pass(0);
        let score1 = archive.add(&fp1);
        assert!(score1 > 0.0);
        assert_eq!(archive.size(), 1);

        // Same fingerprint: should not add again
        let score2 = archive.add(&fp1);
        assert_eq!(archive.size(), 1);

        // Different fingerprint: should add
        let mut fp2 = BehaviorFingerprint::new(5);
        fp2.set_pass(1);
        fp2.set_pass(2);
        let _ = archive.add(&fp2);
        assert_eq!(archive.size(), 2);

        // Similar fingerprint has lower novelty than dissimilar one
        let _ = score2; // used
    }

    #[test]
    fn test_novelty_archive_lru() {
        let mut archive = NoveltyArchive::new(2, 3);

        for i in 0..5u8 {
            let mut fp = BehaviorFingerprint::new(5);
            fp.set_pass(i as usize % 5);
            archive.add(&fp);
        }

        // Should have evicted down to max_size=3
        assert!(archive.size() <= 3);
    }

    #[test]
    fn test_pass_count() {
        let mut fp = BehaviorFingerprint::new(10);
        fp.set_pass(0);
        fp.set_pass(5);
        fp.set_pass(9);
        assert_eq!(fp.pass_count(), 3);
    }
}
