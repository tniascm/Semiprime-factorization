//! Compression-based structure detection in semiprimes.
//!
//! Tests whether compression algorithms can detect structural
//! differences between semiprimes, primes, and random numbers.

use factoring_core::to_base;
use num_bigint::BigUint;
use num_traits::Zero;
use rayon::prelude::*;
use std::collections::HashMap;

/// Run-length encoding compression ratio.
pub fn rle_compression_ratio(data: &[u8]) -> f64 {
    if data.is_empty() {
        return 1.0;
    }
    let mut compressed_len = 0usize;
    let mut i = 0;
    while i < data.len() {
        let val = data[i];
        let mut run = 1;
        while i + run < data.len() && data[i + run] == val && run < 255 {
            run += 1;
        }
        compressed_len += 2; // (count, value)
        i += run;
    }
    compressed_len as f64 / data.len() as f64
}

/// Burrows-Wheeler Transform.
pub fn bwt(data: &[u8]) -> Vec<u8> {
    if data.is_empty() {
        return vec![];
    }
    let n = data.len();
    let mut rotations: Vec<usize> = (0..n).collect();
    rotations.sort_by(|&a, &b| {
        for i in 0..n {
            let ca = data[(a + i) % n];
            let cb = data[(b + i) % n];
            if ca != cb {
                return ca.cmp(&cb);
            }
        }
        std::cmp::Ordering::Equal
    });
    rotations
        .iter()
        .map(|&r| data[(r + n - 1) % n])
        .collect()
}

/// Measure compressibility after BWT (BWT tends to group similar bytes).
pub fn bwt_rle_ratio(data: &[u8]) -> f64 {
    let transformed = bwt(data);
    rle_compression_ratio(&transformed)
}

/// Convert a BigUint to its byte representation for compression analysis.
pub fn number_to_bytes(n: &BigUint) -> Vec<u8> {
    n.to_bytes_be()
}

/// An LZ77 encoded token: (offset into sliding window, match length, literal next byte).
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
struct Lz77Token {
    offset: u16,
    length: u16,
    next: u8,
}

impl Lz77Token {
    /// Encoded size in bytes for this token: 2 (offset) + 2 (length) + 1 (next byte).
    const ENCODED_SIZE: usize = 5;
}

/// Simplified LZ77 compression ratio.
///
/// Scans `data` using a sliding window of `window_size` bytes, finds the longest
/// match in the preceding window, and encodes as (offset, length, next_byte) triples.
/// Returns compressed_size / original_size.
pub fn lz77_compression_ratio(data: &[u8], window_size: usize) -> f64 {
    if data.is_empty() {
        return 1.0;
    }

    let mut tokens: Vec<Lz77Token> = Vec::new();
    let mut pos = 0;

    while pos < data.len() {
        let window_start = if pos > window_size {
            pos - window_size
        } else {
            0
        };

        let mut best_offset: u16 = 0;
        let mut best_length: u16 = 0;

        // Search for the longest match in the window
        for start in window_start..pos {
            let mut match_len: usize = 0;
            let max_match = (data.len() - pos - 1).min(u16::MAX as usize);
            while match_len < max_match && data[start + match_len] == data[pos + match_len] {
                match_len += 1;
            }
            if match_len as u16 > best_length {
                best_length = match_len as u16;
                best_offset = (pos - start) as u16;
            }
        }

        let next_byte_pos = pos + best_length as usize;
        let next_byte = if next_byte_pos < data.len() {
            data[next_byte_pos]
        } else {
            0
        };

        tokens.push(Lz77Token {
            offset: best_offset,
            length: best_length,
            next: next_byte,
        });

        pos += best_length as usize + 1;
    }

    let compressed_size = tokens.len() * Lz77Token::ENCODED_SIZE;
    compressed_size as f64 / data.len() as f64
}

/// Order-1 (conditional) entropy estimate: H(X_i | X_{i-1}).
///
/// Computes conditional entropy using bigram (pair) frequencies. This captures
/// sequential dependencies that byte_entropy (order-0) misses. For data with
/// strong sequential patterns, order-1 entropy will be significantly lower
/// than order-0 entropy.
pub fn order1_entropy(data: &[u8]) -> f64 {
    if data.len() < 2 {
        return 0.0;
    }

    // Count bigram frequencies: how often byte `b` follows byte `a`
    let mut bigram_counts: HashMap<(u8, u8), u64> = HashMap::new();
    // Count unigram frequencies for the conditioning byte
    let mut unigram_counts: HashMap<u8, u64> = HashMap::new();

    for window in data.windows(2) {
        let a = window[0];
        let b = window[1];
        *bigram_counts.entry((a, b)).or_insert(0) += 1;
        *unigram_counts.entry(a).or_insert(0) += 1;
    }

    let total_bigrams = (data.len() - 1) as f64;

    // H(X_i | X_{i-1}) = - sum_{a,b} P(a,b) * log2(P(b|a))
    // where P(b|a) = count(a,b) / count(a)
    let mut entropy = 0.0;
    for (&(a, _b), &count) in &bigram_counts {
        let p_ab = count as f64 / total_bigrams;
        let p_b_given_a = count as f64 / unigram_counts[&a] as f64;
        entropy -= p_ab * p_b_given_a.log2();
    }

    entropy
}

/// Compute the residue pattern of `n` modulo each prime in `primes`.
///
/// Returns a byte vector where each element is `(n mod p_i) as u8`.
/// For primes < 256 the residue fits in a single byte. For larger primes
/// the value is truncated to the low byte, but for the typical small-prime
/// sets used in experiments, this is exact.
pub fn residue_pattern(n: &BigUint, primes: &[u64]) -> Vec<u8> {
    primes
        .iter()
        .map(|&p| {
            let p_big = BigUint::from(p);
            let residue = n % &p_big;
            // Extract low byte. For p < 256 this is the exact residue.
            if residue.is_zero() {
                0u8
            } else {
                let bytes = residue.to_bytes_le();
                bytes[0]
            }
        })
        .collect()
}

/// Results of a differential compression test between semiprimes and random numbers.
#[derive(Debug, Clone)]
pub struct DifferentialResult {
    pub semi_avg_rle: f64,
    pub rand_avg_rle: f64,
    pub semi_avg_lz77: f64,
    pub rand_avg_lz77: f64,
    pub semi_avg_entropy: f64,
    pub rand_avg_entropy: f64,
}

/// Helper: compute the mean of a slice of f64 values.
fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

/// Metrics computed on a single residue pattern.
struct PatternMetrics {
    rle: f64,
    lz77: f64,
    entropy: f64,
}

/// Compute compression metrics for a single number's residue pattern.
fn compute_pattern_metrics(n: &BigUint, primes: &[u64]) -> PatternMetrics {
    let pattern = residue_pattern(n, primes);
    PatternMetrics {
        rle: rle_compression_ratio(&pattern),
        lz77: lz77_compression_ratio(&pattern, 256),
        entropy: order1_entropy(&pattern),
    }
}

/// Differential compression test: computes residue patterns for semiprimes and
/// random numbers, then measures compression ratios (RLE, LZ77, order-1 entropy)
/// for each group.
///
/// Returns a `DifferentialResult` with average metrics for each group.
pub fn differential_compression_test(
    semiprimes: &[BigUint],
    randoms: &[BigUint],
    primes: &[u64],
) -> DifferentialResult {
    let semi_metrics: Vec<PatternMetrics> = semiprimes
        .par_iter()
        .map(|n| compute_pattern_metrics(n, primes))
        .collect();

    let rand_metrics: Vec<PatternMetrics> = randoms
        .par_iter()
        .map(|n| compute_pattern_metrics(n, primes))
        .collect();

    let semi_rle: Vec<f64> = semi_metrics.iter().map(|m| m.rle).collect();
    let semi_lz77: Vec<f64> = semi_metrics.iter().map(|m| m.lz77).collect();
    let semi_entropy: Vec<f64> = semi_metrics.iter().map(|m| m.entropy).collect();

    let rand_rle: Vec<f64> = rand_metrics.iter().map(|m| m.rle).collect();
    let rand_lz77: Vec<f64> = rand_metrics.iter().map(|m| m.lz77).collect();
    let rand_entropy: Vec<f64> = rand_metrics.iter().map(|m| m.entropy).collect();

    DifferentialResult {
        semi_avg_rle: mean(&semi_rle),
        rand_avg_rle: mean(&rand_rle),
        semi_avg_lz77: mean(&semi_lz77),
        rand_avg_lz77: mean(&rand_lz77),
        semi_avg_entropy: mean(&semi_entropy),
        rand_avg_entropy: mean(&rand_entropy),
    }
}

/// Approximate Kolmogorov complexity of a number.
///
/// Tries multiple representations (binary, base-10, base-6, base-30 digits),
/// compresses each with BWT+RLE, and returns the minimum compression ratio
/// across all bases. Lower values indicate more compressible (less complex) numbers.
pub fn approx_kolmogorov(n: &BigUint) -> f64 {
    let bases: &[u32] = &[2, 6, 10, 30];

    bases
        .iter()
        .map(|&base| {
            let digits: Vec<u8> = to_base(n, base).iter().map(|&d| d as u8).collect();
            if digits.is_empty() {
                return f64::INFINITY;
            }
            // For small inputs, BWT is fine. For larger ones, still apply
            // since these are digit representations (typically reasonable size).
            let transformed = bwt(&digits);
            rle_compression_ratio(&transformed)
        })
        .fold(f64::INFINITY, f64::min)
}

/// Shannon entropy of byte distribution.
fn byte_entropy(data: &[u8]) -> f64 {
    let mut counts = [0u64; 256];
    for &b in data {
        counts[b as usize] += 1;
    }
    let total = data.len() as f64;
    counts
        .iter()
        .filter(|&&c| c > 0)
        .map(|&c| {
            let p = c as f64 / total;
            -p * p.log2()
        })
        .sum()
}

/// Analyze compressibility of a number across different representations.
#[derive(Debug, Clone)]
pub struct CompressionAnalysis {
    pub raw_bytes: usize,
    pub rle_ratio: f64,
    pub bwt_rle_ratio: f64,
    pub binary_rle_ratio: f64,
    pub bit_transitions: usize,
    pub byte_entropy: f64,
    pub lz77_ratio: f64,
    pub order1_entropy: f64,
}

/// Full compression analysis of a number.
pub fn analyze_compressibility(n: &BigUint) -> CompressionAnalysis {
    let bytes = number_to_bytes(n);
    let binary_digits: Vec<u8> = to_base(n, 2).iter().map(|&d| d as u8).collect();

    let rle_ratio = rle_compression_ratio(&bytes);
    let bwt_rle_ratio = if bytes.len() < 10000 {
        self::bwt_rle_ratio(&bytes)
    } else {
        f64::NAN // BWT is O(n^2), skip for large inputs
    };

    let binary_rle_ratio = rle_compression_ratio(&binary_digits);

    let bit_transitions = binary_digits
        .windows(2)
        .filter(|w| w[0] != w[1])
        .count();

    let byte_entropy = byte_entropy(&bytes);

    let lz77_ratio = lz77_compression_ratio(&bytes, 256);

    let order1_ent = order1_entropy(&bytes);

    CompressionAnalysis {
        raw_bytes: bytes.len(),
        rle_ratio,
        bwt_rle_ratio,
        binary_rle_ratio,
        bit_transitions,
        byte_entropy,
        lz77_ratio,
        order1_entropy: order1_ent,
    }
}

/// Compare compression characteristics of semiprimes vs random numbers.
pub fn batch_comparison(
    semiprimes: &[BigUint],
    randoms: &[BigUint],
) -> (Vec<CompressionAnalysis>, Vec<CompressionAnalysis>) {
    let semi_analyses: Vec<CompressionAnalysis> = semiprimes
        .par_iter()
        .map(|n| analyze_compressibility(n))
        .collect();

    let rand_analyses: Vec<CompressionAnalysis> = randoms
        .par_iter()
        .map(|n| analyze_compressibility(n))
        .collect();

    (semi_analyses, rand_analyses)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rle() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let ratio = rle_compression_ratio(&data);
        assert!(ratio > 0.0 && ratio <= 2.0);
    }

    #[test]
    fn test_bwt() {
        let data = b"banana";
        let result = bwt(data);
        assert_eq!(result.len(), data.len());
    }

    #[test]
    fn test_analyze() {
        let n = BigUint::from(8051u64);
        let analysis = analyze_compressibility(&n);
        assert!(analysis.byte_entropy >= 0.0);
        // Verify new fields are populated
        assert!(analysis.lz77_ratio > 0.0);
        assert!(analysis.order1_entropy >= 0.0 || analysis.order1_entropy.is_nan());
    }

    #[test]
    fn test_lz77() {
        // Repeating pattern: highly compressible data should have ratio < 1.0
        // when the data is long enough relative to the token overhead
        let mut data = Vec::new();
        for _ in 0..200 {
            data.extend_from_slice(b"ABCABC");
        }
        let ratio = lz77_compression_ratio(&data, 256);
        assert!(
            ratio < 1.0,
            "LZ77 ratio on repeating data should be < 1.0, got {}",
            ratio
        );
    }

    #[test]
    fn test_order1_entropy() {
        // Sequential data: very predictable, low order-1 entropy
        let sequential: Vec<u8> = (0..=255).cycle().take(1024).collect();
        let sequential_entropy = order1_entropy(&sequential);

        // Random-ish data: high order-1 entropy
        // Use a simple deterministic pseudo-random sequence
        let mut rng_data = vec![0u8; 1024];
        let mut state: u32 = 12345;
        for byte in rng_data.iter_mut() {
            state = state.wrapping_mul(1103515245).wrapping_add(12345);
            *byte = (state >> 16) as u8;
        }
        let random_entropy = order1_entropy(&rng_data);

        assert!(
            sequential_entropy < random_entropy,
            "Sequential data entropy ({}) should be less than random data entropy ({})",
            sequential_entropy,
            random_entropy
        );
    }

    #[test]
    fn test_residue_pattern() {
        let n = BigUint::from(1000003u64);
        let primes = vec![2, 3, 5, 7, 11, 13, 17, 19, 23, 29];
        let pattern = residue_pattern(&n, &primes);
        assert_eq!(
            pattern.len(),
            primes.len(),
            "Residue pattern length should match number of primes"
        );
        // Verify a known residue: 1000003 mod 2 = 1
        assert_eq!(pattern[0], 1);
        // 1000003 mod 3: 1000003 / 3 = 333334 remainder 1
        assert_eq!(pattern[1], 1);
    }

    #[test]
    fn test_approx_kolmogorov() {
        let n = BigUint::from(123456789u64);
        let k = approx_kolmogorov(&n);
        assert!(
            k.is_finite() && k > 0.0,
            "Kolmogorov approximation should be positive and finite, got {}",
            k
        );
    }
}
