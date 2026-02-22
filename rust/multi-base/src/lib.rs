//! Multi-base number representations for pattern analysis.
//!
//! Explores how representing semiprimes in different number bases
//! exposes structural patterns related to their factors.

use factoring_core::{entropy, to_base, to_rns};
use num_bigint::BigUint;
use rayon::prelude::*;
use std::collections::HashMap;

/// Predefined bases of interest for factorization analysis.
pub const ANALYSIS_BASES: &[(u32, &str)] = &[
    (2, "binary"),
    (3, "ternary"),
    (6, "base-6 (2×3 absorbed)"),
    (10, "decimal"),
    (16, "hexadecimal"),
    (30, "primorial-3 (2×3×5)"),
    (210, "primorial-4 (2×3×5×7)"),
    (41, "prime-41"),
    (43, "prime-43"),
];

/// Default RNS moduli — first 20 primes.
pub const RNS_MODULI: &[u64] = &[
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
];

/// Complete multi-base analysis of a number.
#[derive(Debug, Clone)]
pub struct MultiBaseAnalysis {
    pub n: BigUint,
    pub representations: Vec<BaseRepresentation>,
    pub rns: Vec<u64>,
    pub cross_base_features: CrossBaseFeatures,
}

/// Representation of a number in a specific base with derived statistics.
#[derive(Debug, Clone)]
pub struct BaseRepresentation {
    pub base: u32,
    pub base_name: String,
    pub digits: Vec<u32>,
    pub num_digits: usize,
    pub entropy: f64,
    pub digit_frequencies: HashMap<u32, usize>,
    pub autocorrelation_lag1: f64,
}

/// Features derived from comparing representations across bases.
#[derive(Debug, Clone)]
pub struct CrossBaseFeatures {
    pub entropy_variance: f64,
    pub min_entropy_base: u32,
    pub max_entropy_base: u32,
    pub rns_zero_count: usize,
}

/// Analyze a number across all predefined bases.
pub fn analyze(n: &BigUint) -> MultiBaseAnalysis {
    let representations: Vec<BaseRepresentation> = ANALYSIS_BASES
        .par_iter()
        .map(|&(base, name)| {
            let digits = to_base(n, base);
            let ent = entropy(&digits, base);
            let freqs = digit_frequencies(&digits);
            let autocorr = autocorrelation_lag1(&digits);

            BaseRepresentation {
                base,
                base_name: name.to_string(),
                digits: digits.clone(),
                num_digits: digits.len(),
                entropy: ent,
                digit_frequencies: freqs,
                autocorrelation_lag1: autocorr,
            }
        })
        .collect();

    let rns = to_rns(n, RNS_MODULI);

    let cross_base_features = compute_cross_base_features(&representations, &rns);

    MultiBaseAnalysis {
        n: n.clone(),
        representations,
        rns,
        cross_base_features,
    }
}

/// Compute digit frequency distribution.
fn digit_frequencies(digits: &[u32]) -> HashMap<u32, usize> {
    let mut freq = HashMap::new();
    for &d in digits {
        *freq.entry(d).or_insert(0) += 1;
    }
    freq
}

/// Compute lag-1 autocorrelation of digit sequence.
fn autocorrelation_lag1(digits: &[u32]) -> f64 {
    if digits.len() < 2 {
        return 0.0;
    }
    let mean: f64 = digits.iter().map(|&d| d as f64).sum::<f64>() / digits.len() as f64;
    let variance: f64 =
        digits.iter().map(|&d| (d as f64 - mean).powi(2)).sum::<f64>() / digits.len() as f64;

    if variance == 0.0 {
        return 0.0;
    }

    let covariance: f64 = digits
        .windows(2)
        .map(|w| (w[0] as f64 - mean) * (w[1] as f64 - mean))
        .sum::<f64>()
        / (digits.len() - 1) as f64;

    covariance / variance
}

/// Derive cross-base comparison features.
fn compute_cross_base_features(reps: &[BaseRepresentation], rns: &[u64]) -> CrossBaseFeatures {
    let entropies: Vec<f64> = reps.iter().map(|r| r.entropy).collect();
    let mean_ent: f64 = entropies.iter().sum::<f64>() / entropies.len() as f64;
    let entropy_variance =
        entropies.iter().map(|e| (e - mean_ent).powi(2)).sum::<f64>() / entropies.len() as f64;

    let min_entropy_base = reps
        .iter()
        .min_by(|a, b| a.entropy.partial_cmp(&b.entropy).unwrap())
        .map(|r| r.base)
        .unwrap_or(2);

    let max_entropy_base = reps
        .iter()
        .max_by(|a, b| a.entropy.partial_cmp(&b.entropy).unwrap())
        .map(|r| r.base)
        .unwrap_or(2);

    let rns_zero_count = rns.iter().filter(|&&r| r == 0).count();

    CrossBaseFeatures {
        entropy_variance,
        min_entropy_base,
        max_entropy_base,
        rns_zero_count,
    }
}

/// Compare multi-base analysis of a semiprime vs a random number of same size.
pub fn compare_semiprime_vs_random(
    semiprime: &MultiBaseAnalysis,
    random: &MultiBaseAnalysis,
) -> Vec<(String, f64, f64)> {
    semiprime
        .representations
        .iter()
        .zip(random.representations.iter())
        .map(|(s, r)| {
            (
                s.base_name.clone(),
                s.entropy,
                r.entropy,
            )
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Balanced Ternary Representation
// ---------------------------------------------------------------------------

/// Statistics for a balanced ternary representation.
#[derive(Debug, Clone)]
pub struct BalancedTernaryStats {
    /// Total number of balanced ternary digits.
    pub num_digits: usize,
    /// Count of zero digits.
    pub num_zeros: usize,
    /// Count of +1 digits.
    pub num_positive: usize,
    /// Count of -1 digits.
    pub num_negative: usize,
    /// Non-zero digit count (analogous to Hamming weight).
    pub weight: usize,
}

/// Convert a `BigUint` to balanced ternary representation (digits -1, 0, 1 stored as `i8`).
///
/// Returns digits in big-endian order (most significant position first).
/// Algorithm: compute regular base-3 digits, then propagate carries so every
/// digit falls in {-1, 0, 1}.
pub fn to_balanced_ternary(n: &BigUint) -> Vec<i8> {
    use num_traits::Zero;

    if n.is_zero() {
        return vec![0];
    }

    // Step 1: get regular base-3 digits in little-endian order.
    let base3_big_endian = to_base(n, 3);
    let mut digits: Vec<i8> = base3_big_endian
        .iter()
        .rev()
        .map(|&d| d as i8)
        .collect();

    // Step 2: carry propagation in little-endian order.
    let mut i = 0;
    while i < digits.len() {
        if digits[i] > 1 {
            // digit is 2 or 3 (3 can appear from a carry into a position that was 2)
            let carry = if digits[i] == 2 {
                digits[i] = -1;
                1
            } else {
                // digits[i] == 3
                digits[i] = 0;
                1
            };
            if i + 1 < digits.len() {
                digits[i + 1] += carry;
            } else {
                digits.push(carry);
            }
        }
        i += 1;
    }

    // Remove trailing zeros in little-endian (leading zeros in big-endian).
    while digits.len() > 1 && digits.last() == Some(&0) {
        digits.pop();
    }

    // Reverse to big-endian.
    digits.reverse();
    digits
}

/// Compute statistics for a balanced ternary digit sequence.
pub fn balanced_ternary_stats(digits: &[i8]) -> BalancedTernaryStats {
    let num_zeros = digits.iter().filter(|&&d| d == 0).count();
    let num_positive = digits.iter().filter(|&&d| d == 1).count();
    let num_negative = digits.iter().filter(|&&d| d == -1).count();

    BalancedTernaryStats {
        num_digits: digits.len(),
        num_zeros,
        num_positive,
        num_negative,
        weight: num_positive + num_negative,
    }
}

// ---------------------------------------------------------------------------
// Factorial Number System
// ---------------------------------------------------------------------------

/// Statistics for a factorial number system representation.
#[derive(Debug, Clone)]
pub struct FactorialBaseStats {
    /// Total number of factorial-base digits.
    pub num_digits: usize,
    /// Maximum ratio digit_i / (i+1) across all positions. Measures how "full"
    /// each position is (1.0 means the digit is at its maximum allowed value).
    pub max_digit_ratio: f64,
    /// Count of zero digits.
    pub zero_count: usize,
}

/// Convert a `BigUint` to the factorial number system.
///
/// Position 0 has radix 2 (digit 0 or 1), position 1 has radix 3 (0..2), etc.
/// Returns digits in positional order starting from position 0 (little-endian
/// by position).
///
/// Algorithm: repeatedly divide by 2, 3, 4, ... collecting remainders.
pub fn to_factorial_base(n: &BigUint) -> Vec<u32> {
    use num_traits::Zero;

    if n.is_zero() {
        return vec![0];
    }

    let mut digits = Vec::new();
    let mut remaining = n.clone();
    let mut radix = 2u32;

    while !remaining.is_zero() {
        let radix_big = BigUint::from(radix);
        let digit = &remaining % &radix_big;
        digits.push(digit.to_u32_digits().first().copied().unwrap_or(0));
        remaining /= &radix_big;
        radix += 1;
    }

    digits
}

/// Compute statistics for a factorial-base digit sequence.
///
/// `digits` is expected in positional order (position 0 first), where position
/// `i` has radix `i + 2` and maximum digit value `i + 1`.
pub fn factorial_base_stats(digits: &[u32]) -> FactorialBaseStats {
    let mut max_digit_ratio: f64 = 0.0;
    let mut zero_count: usize = 0;

    for (i, &d) in digits.iter().enumerate() {
        let max_val = (i + 1) as f64; // max allowed digit at position i is i+1
        if max_val > 0.0 {
            let ratio = d as f64 / max_val;
            if ratio > max_digit_ratio {
                max_digit_ratio = ratio;
            }
        }
        if d == 0 {
            zero_count += 1;
        }
    }

    FactorialBaseStats {
        num_digits: digits.len(),
        max_digit_ratio,
        zero_count,
    }
}

// ---------------------------------------------------------------------------
// Cross-base Anomaly Detector
// ---------------------------------------------------------------------------

/// A detected anomaly indicating a base where semiprimes behave differently
/// from random numbers.
#[derive(Debug, Clone)]
pub struct BaseAnomaly {
    /// The numeric base where the anomaly was detected.
    pub base: u32,
    /// Human-readable base name.
    pub base_name: String,
    /// Z-score: how many standard deviations the semiprime mean differs from
    /// the random mean. Positive means semiprimes have higher values.
    pub z_score: f64,
    /// Which metric the anomaly was detected in (e.g. "entropy", "autocorrelation").
    pub metric: String,
}

/// Detect cross-base anomalies by comparing semiprime analyses against random
/// number analyses.
///
/// For each base, computes the z-score of the semiprime mean relative to the
/// random distribution for both entropy and autocorrelation. Returns anomalies
/// sorted by |z_score| descending — the top entries are bases where semiprimes
/// behave most differently from random numbers.
pub fn cross_base_anomaly_score(
    semiprime_analyses: &[MultiBaseAnalysis],
    random_analyses: &[MultiBaseAnalysis],
) -> Vec<BaseAnomaly> {
    if semiprime_analyses.is_empty() || random_analyses.is_empty() {
        return Vec::new();
    }

    // Determine how many base representations each analysis has.
    let num_bases = semiprime_analyses[0].representations.len();
    let mut anomalies = Vec::new();

    for base_idx in 0..num_bases {
        let base = semiprime_analyses[0].representations[base_idx].base;
        let base_name = semiprime_analyses[0].representations[base_idx]
            .base_name
            .clone();

        // --- Entropy anomaly ---
        let random_entropies: Vec<f64> = random_analyses
            .iter()
            .filter(|a| a.representations.len() > base_idx)
            .map(|a| a.representations[base_idx].entropy)
            .collect();

        let semiprime_entropies: Vec<f64> = semiprime_analyses
            .iter()
            .filter(|a| a.representations.len() > base_idx)
            .map(|a| a.representations[base_idx].entropy)
            .collect();

        if let Some(anomaly) = compute_z_score_anomaly(
            &semiprime_entropies,
            &random_entropies,
            base,
            &base_name,
            "entropy",
        ) {
            anomalies.push(anomaly);
        }

        // --- Autocorrelation anomaly ---
        let random_autocorrs: Vec<f64> = random_analyses
            .iter()
            .filter(|a| a.representations.len() > base_idx)
            .map(|a| a.representations[base_idx].autocorrelation_lag1)
            .collect();

        let semiprime_autocorrs: Vec<f64> = semiprime_analyses
            .iter()
            .filter(|a| a.representations.len() > base_idx)
            .map(|a| a.representations[base_idx].autocorrelation_lag1)
            .collect();

        if let Some(anomaly) = compute_z_score_anomaly(
            &semiprime_autocorrs,
            &random_autocorrs,
            base,
            &base_name,
            "autocorrelation",
        ) {
            anomalies.push(anomaly);
        }
    }

    // Sort by |z_score| descending.
    anomalies.sort_by(|a, b| {
        b.z_score
            .abs()
            .partial_cmp(&a.z_score.abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    anomalies
}

/// Helper: compute z-score for a set of semiprime values relative to a random
/// distribution. Returns `None` if the random stddev is zero or insufficient data.
fn compute_z_score_anomaly(
    semiprime_values: &[f64],
    random_values: &[f64],
    base: u32,
    base_name: &str,
    metric: &str,
) -> Option<BaseAnomaly> {
    if random_values.len() < 2 || semiprime_values.is_empty() {
        return None;
    }

    let random_mean = random_values.iter().sum::<f64>() / random_values.len() as f64;
    let random_variance = random_values
        .iter()
        .map(|v| (v - random_mean).powi(2))
        .sum::<f64>()
        / random_values.len() as f64;
    let random_stddev = random_variance.sqrt();

    if random_stddev < 1e-12 {
        return None;
    }

    let semiprime_mean = semiprime_values.iter().sum::<f64>() / semiprime_values.len() as f64;
    let z_score = (semiprime_mean - random_mean) / random_stddev;

    Some(BaseAnomaly {
        base,
        base_name: base_name.to_string(),
        z_score,
        metric: metric.to_string(),
    })
}

// ---------------------------------------------------------------------------
// Digit Transition Matrix
// ---------------------------------------------------------------------------

/// Compute a digit transition matrix for a digit sequence in the given base.
///
/// Returns a `base × base` matrix where entry `(i, j)` is the empirical
/// probability P(next_digit = j | current_digit = i). Rows that have no
/// observed transitions (the digit never appears except possibly at the end)
/// are left as all zeros.
pub fn digit_transition_matrix(digits: &[u32], base: u32) -> Vec<Vec<f64>> {
    let b = base as usize;
    let mut counts = vec![vec![0u64; b]; b];

    for window in digits.windows(2) {
        let from = window[0] as usize;
        let to = window[1] as usize;
        if from < b && to < b {
            counts[from][to] += 1;
        }
    }

    // Normalize each row to get probabilities.
    counts
        .iter()
        .map(|row| {
            let row_sum: u64 = row.iter().sum();
            if row_sum == 0 {
                vec![0.0; b]
            } else {
                row.iter().map(|&c| c as f64 / row_sum as f64).collect()
            }
        })
        .collect()
}

/// Compute the average conditional entropy of a row-stochastic transition matrix.
///
/// For each row (conditioning on the current digit), computes Shannon entropy
/// of the transition probabilities, then averages across all rows that have
/// at least one transition.
pub fn transition_matrix_entropy(matrix: &[Vec<f64>]) -> f64 {
    let mut total_entropy = 0.0;
    let mut active_rows = 0usize;

    for row in matrix {
        let row_sum: f64 = row.iter().sum();
        if row_sum < 1e-12 {
            continue; // skip rows with no observed transitions
        }

        let row_entropy: f64 = row
            .iter()
            .filter(|&&p| p > 0.0)
            .map(|&p| -p * p.log2())
            .sum();

        total_entropy += row_entropy;
        active_rows += 1;
    }

    if active_rows == 0 {
        0.0
    } else {
        total_entropy / active_rows as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_analyze_small() {
        let n = BigUint::from(8051u32); // 83 × 97
        let analysis = analyze(&n);
        assert!(!analysis.representations.is_empty());
        assert_eq!(analysis.rns.len(), RNS_MODULI.len());
    }

    #[test]
    fn test_rns_detects_factors() {
        let n = BigUint::from(15u32); // 3 × 5
        let rns = to_rns(&n, RNS_MODULI);
        // N mod 3 = 0, N mod 5 = 0
        assert_eq!(rns[1], 0); // mod 3
        assert_eq!(rns[2], 0); // mod 5
    }

    #[test]
    fn test_entropy_ranges() {
        let n = BigUint::from(1_000_003u64 * 1_000_033u64);
        let analysis = analyze(&n);
        for rep in &analysis.representations {
            assert!(rep.entropy >= 0.0);
        }
    }

    #[test]
    fn test_balanced_ternary() {
        // 10 = 1*9 + 0*3 + 1*1 → [1, 0, 1]
        let digits_10 = to_balanced_ternary(&BigUint::from(10u32));
        assert_eq!(digits_10, vec![1i8, 0, 1]);

        // 5 = 1*9 - 1*3 - 1*1 → [1, -1, -1]
        let digits_5 = to_balanced_ternary(&BigUint::from(5u32));
        assert_eq!(digits_5, vec![1i8, -1, -1]);

        // 0 → [0]
        let digits_0 = to_balanced_ternary(&BigUint::from(0u32));
        assert_eq!(digits_0, vec![0i8]);

        // Verify stats for 5: weight=3, num_negative=2, num_positive=1, num_zeros=0
        let stats = balanced_ternary_stats(&digits_5);
        assert_eq!(stats.num_digits, 3);
        assert_eq!(stats.num_zeros, 0);
        assert_eq!(stats.num_positive, 1);
        assert_eq!(stats.num_negative, 2);
        assert_eq!(stats.weight, 3);
    }

    #[test]
    fn test_factorial_base() {
        // 5 / 2 = 2 rem 1, 2 / 3 = 0 rem 2 → [1, 2]
        let digits_5 = to_factorial_base(&BigUint::from(5u32));
        assert_eq!(digits_5, vec![1, 2]);

        // Verify: 1*1! + 2*2! = 1 + 4 = 5 ✓

        // 0 → [0]
        let digits_0 = to_factorial_base(&BigUint::from(0u32));
        assert_eq!(digits_0, vec![0]);

        // Stats for [1, 2]: max_digit_ratio = max(1/1, 2/2) = 1.0, zero_count = 0
        let stats = factorial_base_stats(&digits_5);
        assert_eq!(stats.num_digits, 2);
        assert!((stats.max_digit_ratio - 1.0).abs() < 1e-10);
        assert_eq!(stats.zero_count, 0);
    }

    #[test]
    fn test_cross_base_anomaly() {
        // Create mock analyses with controlled entropy/autocorrelation values.
        let make_analysis = |entropy_vals: Vec<f64>, autocorr_vals: Vec<f64>| -> MultiBaseAnalysis {
            let reps: Vec<BaseRepresentation> = ANALYSIS_BASES
                .iter()
                .enumerate()
                .map(|(i, &(base, name))| BaseRepresentation {
                    base,
                    base_name: name.to_string(),
                    digits: vec![0],
                    num_digits: 1,
                    entropy: *entropy_vals.get(i).unwrap_or(&0.5),
                    digit_frequencies: HashMap::new(),
                    autocorrelation_lag1: *autocorr_vals.get(i).unwrap_or(&0.0),
                })
                .collect();
            MultiBaseAnalysis {
                n: BigUint::from(1u32),
                representations: reps,
                rns: vec![],
                cross_base_features: CrossBaseFeatures {
                    entropy_variance: 0.0,
                    min_entropy_base: 2,
                    max_entropy_base: 2,
                    rns_zero_count: 0,
                },
            }
        };

        let num_bases = ANALYSIS_BASES.len();

        // Semiprimes: high entropy in base 0 (binary), low elsewhere.
        let sp_entropy = {
            let mut v = vec![0.5; num_bases];
            v[0] = 0.95; // high entropy in binary
            v
        };
        let sp_autocorr = vec![0.0; num_bases];
        let sp = vec![
            make_analysis(sp_entropy.clone(), sp_autocorr.clone()),
            make_analysis(sp_entropy.clone(), sp_autocorr.clone()),
        ];

        // Randoms: moderate entropy everywhere with some variance.
        let r1_entropy = vec![0.5; num_bases];
        let r2_entropy = vec![0.6; num_bases];
        let r_autocorr = vec![0.0; num_bases];
        let randoms = vec![
            make_analysis(r1_entropy, r_autocorr.clone()),
            make_analysis(r2_entropy, r_autocorr),
        ];

        let anomalies = cross_base_anomaly_score(&sp, &randoms);

        // Should have some anomalies computed.
        assert!(!anomalies.is_empty());

        // The top anomaly should be for binary (base 2) entropy, since that's
        // where semiprimes differ most from random.
        let top = &anomalies[0];
        assert_eq!(top.base, 2);
        assert_eq!(top.metric, "entropy");
        assert!(top.z_score.abs() > 1.0);

        // Verify sorted by |z_score| descending.
        for w in anomalies.windows(2) {
            assert!(w[0].z_score.abs() >= w[1].z_score.abs());
        }
    }

    #[test]
    fn test_transition_matrix() {
        // Binary digit sequence.
        let digits = vec![0, 1, 0, 1, 1, 0, 0, 1];
        let matrix = digit_transition_matrix(&digits, 2);

        // Matrix should be 2×2.
        assert_eq!(matrix.len(), 2);
        assert_eq!(matrix[0].len(), 2);
        assert_eq!(matrix[1].len(), 2);

        // Verify row-stochastic: each row sums to 1 (if it has transitions).
        for row in &matrix {
            let row_sum: f64 = row.iter().sum();
            if row_sum > 0.0 {
                assert!(
                    (row_sum - 1.0).abs() < 1e-10,
                    "Row does not sum to 1: sum = {}",
                    row_sum
                );
            }
        }

        // Verify transition counts for digits [0,1,0,1,1,0,0,1]:
        // 0→1: 3 times, 0→0: 1 time  → P(1|0) = 3/4, P(0|0) = 1/4
        // 1→0: 2 times, 1→1: 1 time  → P(0|1) = 2/3, P(1|1) = 1/3
        assert!((matrix[0][0] - 0.25).abs() < 1e-10);
        assert!((matrix[0][1] - 0.75).abs() < 1e-10);
        assert!((matrix[1][0] - 2.0 / 3.0).abs() < 1e-10);
        assert!((matrix[1][1] - 1.0 / 3.0).abs() < 1e-10);

        // Verify transition_matrix_entropy returns a reasonable value.
        let te = transition_matrix_entropy(&matrix);
        assert!(te > 0.0, "Transition entropy should be positive");
        // Maximum entropy for binary transitions is 1.0 bit per row.
        assert!(te <= 1.0 + 1e-10, "Transition entropy too high: {}", te);
    }
}
