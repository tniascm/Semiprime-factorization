//! Analysis of discriminant structure for Eichler-Selberg applied to N = pq.
//!
//! The key experimental question: for discriminants D = t^2 - 4l arising from
//! the Eichler-Selberg trace formula applied to N = pq (semiprime), do the
//! class numbers H(D) exhibit structure correlated with the factorization?
//!
//! Specifically:
//! 1. Do discriminants split differently mod p vs mod q?
//! 2. Does the splitting pattern of D in Z_p and Z_q correlate with H(D)?
//! 3. Can we exploit the Kronecker symbol (D|p) and (D|q) to speed up H(D)?
//! 4. Is there any multiplicativity: H(D) ~ f(D mod p-stuff) * g(D mod q-stuff)?

use crate::class_number::{class_number_exact, kronecker_symbol};

/// Full analysis of discriminant structure for a specific (N, l) pair.
#[derive(Debug, Clone)]
pub struct DiscriminantAnalysis {
    /// The level N = p * q.
    pub n: u64,
    /// Known factor p (for analysis purposes).
    pub p: u64,
    /// Known factor q (for analysis purposes).
    pub q: u64,
    /// The Hecke operator index l.
    pub l: u64,
    /// Detailed info for each discriminant.
    pub discriminants: Vec<DiscriminantInfo>,
    /// Aggregate splitting pattern statistics.
    pub splitting_pattern: SplittingPattern,
}

/// Detailed information about a single discriminant D = t^2 - 4l.
#[derive(Debug, Clone)]
pub struct DiscriminantInfo {
    /// The parameter t such that D = t^2 - 4l.
    pub t: i64,
    /// The discriminant D = t^2 - 4l.
    pub d: i64,
    /// The class number h(D) (exact, via form counting).
    pub h_d: u64,
    /// Kronecker symbol (D|p).
    pub kronecker_p: i64,
    /// Kronecker symbol (D|q).
    pub kronecker_q: i64,
    /// Whether D is a quadratic residue mod p (i.e., (D|p) = 1).
    pub splits_in_zp: bool,
    /// Whether D is a quadratic residue mod q (i.e., (D|q) = 1).
    pub splits_in_zq: bool,
}

/// Aggregate statistics about the splitting pattern of discriminants.
#[derive(Debug, Clone)]
pub struct SplittingPattern {
    /// Total number of discriminants analyzed.
    pub total: usize,
    /// Number of D that split in both Z_p and Z_q.
    pub split_both: usize,
    /// Number of D that split in Z_p only.
    pub split_p_only: usize,
    /// Number of D that split in Z_q only.
    pub split_q_only: usize,
    /// Number of D that split in neither.
    pub split_neither: usize,
    /// Sum of H(D) for discriminants splitting in both.
    pub h_sum_split_both: f64,
    /// Sum of H(D) for discriminants splitting in Z_p only.
    pub h_sum_split_p: f64,
    /// Sum of H(D) for discriminants splitting in Z_q only.
    pub h_sum_split_q: f64,
    /// Sum of H(D) for discriminants splitting in neither.
    pub h_sum_split_neither: f64,
}

/// Result of testing multiplicativity of H(D) with respect to the factorization.
#[derive(Debug, Clone)]
pub struct MultiplicativityResult {
    /// The level N = p * q.
    pub n: u64,
    pub p: u64,
    pub q: u64,
    /// For each discriminant D, the tuple (D, H(D), D mod p, D mod q).
    pub data_points: Vec<MultiplicativityDataPoint>,
    /// Average relative error of the multiplicative approximation.
    pub avg_relative_error: f64,
    /// Maximum relative error.
    pub max_relative_error: f64,
    /// Whether the data suggests multiplicativity (error < threshold).
    pub suggests_multiplicativity: bool,
}

/// A single data point for the multiplicativity test.
#[derive(Debug, Clone)]
pub struct MultiplicativityDataPoint {
    pub d: i64,
    pub h_d: u64,
    pub d_mod_p: i64,
    pub d_mod_q: i64,
    pub kronecker_p: i64,
    pub kronecker_q: i64,
    pub kronecker_n: i64,
}

/// Analyze the discriminant structure for a specific (N=pq, l) configuration.
///
/// For each t with |t| < 2*sqrt(l), computes D = t^2 - 4l and analyzes:
/// - The class number h(D)
/// - The Kronecker symbols (D|p) and (D|q)
/// - Whether D splits in Z_p and Z_q
pub fn analyze_discriminants(n: u64, p: u64, q: u64, l: u64) -> DiscriminantAnalysis {
    assert_eq!(n, p * q, "N must equal p * q");

    let bound = (2.0 * (l as f64).sqrt()).floor() as i64;
    let mut disc_infos = Vec::new();

    for t in (-bound)..=bound {
        let d = (t as i64) * (t as i64) - 4 * (l as i64);
        if d >= 0 {
            continue;
        }

        // Check valid discriminant
        let d_mod_4 = ((d % 4) + 4) % 4;
        if d_mod_4 != 0 && d_mod_4 != 1 {
            continue;
        }

        let h_d = class_number_exact(d);
        let kp = kronecker_symbol(d, p);
        let kq = kronecker_symbol(d, q);

        disc_infos.push(DiscriminantInfo {
            t,
            d,
            h_d,
            kronecker_p: kp,
            kronecker_q: kq,
            splits_in_zp: kp == 1,
            splits_in_zq: kq == 1,
        });
    }

    // Compute splitting pattern statistics
    let total = disc_infos.len();
    let mut split_both = 0usize;
    let mut split_p_only = 0usize;
    let mut split_q_only = 0usize;
    let mut split_neither = 0usize;
    let mut h_sum_both = 0.0f64;
    let mut h_sum_p = 0.0f64;
    let mut h_sum_q = 0.0f64;
    let mut h_sum_neither = 0.0f64;

    for info in &disc_infos {
        let h = info.h_d as f64;
        match (info.splits_in_zp, info.splits_in_zq) {
            (true, true) => {
                split_both += 1;
                h_sum_both += h;
            }
            (true, false) => {
                split_p_only += 1;
                h_sum_p += h;
            }
            (false, true) => {
                split_q_only += 1;
                h_sum_q += h;
            }
            (false, false) => {
                split_neither += 1;
                h_sum_neither += h;
            }
        }
    }

    let splitting_pattern = SplittingPattern {
        total,
        split_both,
        split_p_only,
        split_q_only,
        split_neither,
        h_sum_split_both: h_sum_both,
        h_sum_split_p: h_sum_p,
        h_sum_split_q: h_sum_q,
        h_sum_split_neither: h_sum_neither,
    };

    DiscriminantAnalysis {
        n,
        p,
        q,
        l,
        discriminants: disc_infos,
        splitting_pattern,
    }
}

/// Analyze discriminants across multiple l values.
///
/// This gives a broader picture of how the splitting pattern changes
/// as l varies, which is important for understanding whether the
/// structure can be exploited for factoring.
pub fn analyze_discriminants_multi_l(
    n: u64,
    p: u64,
    q: u64,
    l_values: &[u64],
) -> Vec<DiscriminantAnalysis> {
    l_values
        .iter()
        .map(|&l| analyze_discriminants(n, p, q, l))
        .collect()
}

/// Test whether H(D) exhibits multiplicativity with respect to p and q.
///
/// For N = pq, we test whether the class number H(D) can be approximated
/// as a function of (D mod p) and (D mod q) separately.
///
/// Specifically, we test whether:
///   (D|N) = (D|p) * (D|q)  (this is always true by multiplicativity of Kronecker)
///
/// The deeper question: does H(D) decompose in a way related to the
/// splitting behavior mod p and mod q? E.g., is the average H(D) for
/// discriminants that split in Z_p different from those that don't?
pub fn check_class_number_multiplicativity(
    n: u64,
    p: u64,
    q: u64,
) -> MultiplicativityResult {
    assert_eq!(n, p * q, "N must equal p * q");

    let mut data_points = Vec::new();

    // Gather data from several l values
    for l in [2u64, 3, 5, 7, 11, 13] {
        if n % l == 0 {
            continue;
        }

        let bound = (2.0 * (l as f64).sqrt()).floor() as i64;

        for t in (-bound)..=bound {
            let d = (t as i64) * (t as i64) - 4 * (l as i64);
            if d >= 0 {
                continue;
            }

            let d_mod_4 = ((d % 4) + 4) % 4;
            if d_mod_4 != 0 && d_mod_4 != 1 {
                continue;
            }

            let h_d = class_number_exact(d);
            let kp = kronecker_symbol(d, p);
            let kq = kronecker_symbol(d, q);
            let kn = kronecker_symbol(d, n);

            data_points.push(MultiplicativityDataPoint {
                d,
                h_d,
                d_mod_p: ((d % (p as i64)) + (p as i64)) % (p as i64),
                d_mod_q: ((d % (q as i64)) + (q as i64)) % (q as i64),
                kronecker_p: kp,
                kronecker_q: kq,
                kronecker_n: kn,
            });
        }
    }

    // Test Kronecker multiplicativity: (D|N) should equal (D|p) * (D|q)
    // This is always true by the Kronecker symbol definition -- it's a sanity check.
    let mut mult_errors = Vec::new();
    for dp in &data_points {
        let product = dp.kronecker_p * dp.kronecker_q;
        if product != dp.kronecker_n {
            // This should never happen for Kronecker symbols
            mult_errors.push(dp.d);
        }
    }

    // Compute average class number by splitting category
    let mut h_by_category: std::collections::HashMap<(i64, i64), Vec<u64>> =
        std::collections::HashMap::new();

    for dp in &data_points {
        h_by_category
            .entry((dp.kronecker_p, dp.kronecker_q))
            .or_default()
            .push(dp.h_d);
    }

    // Try a simple multiplicative model:
    // H_model(D) = alpha * (1 + beta * (D|p)) * (1 + gamma * (D|q))
    // Fit by least squares on the data.
    // For now, just report the error of assuming H is constant within each category.

    let overall_avg: f64 = if data_points.is_empty() {
        1.0
    } else {
        data_points.iter().map(|dp| dp.h_d as f64).sum::<f64>() / data_points.len() as f64
    };

    let mut total_rel_error = 0.0f64;
    let mut max_rel_error = 0.0f64;

    for dp in &data_points {
        // Use category average as prediction
        let category = (dp.kronecker_p, dp.kronecker_q);
        let cat_avg: f64 = match h_by_category.get(&category) {
            Some(vals) => vals.iter().map(|&v| v as f64).sum::<f64>() / vals.len() as f64,
            None => overall_avg,
        };

        let actual = dp.h_d as f64;
        if actual > 0.0 {
            let rel_error = (actual - cat_avg).abs() / actual;
            total_rel_error += rel_error;
            if rel_error > max_rel_error {
                max_rel_error = rel_error;
            }
        }
    }

    let avg_relative_error = if data_points.is_empty() {
        0.0
    } else {
        total_rel_error / data_points.len() as f64
    };

    MultiplicativityResult {
        n,
        p,
        q,
        data_points,
        avg_relative_error,
        max_relative_error: max_rel_error,
        suggests_multiplicativity: avg_relative_error < 0.1, // Threshold
    }
}

/// Analyze the distribution of Kronecker symbols (D|p) across Eichler-Selberg discriminants.
///
/// By quadratic reciprocity, for D = t^2 - 4l:
///   (D|p) depends on D mod p, which is (t^2 - 4l) mod p.
///   As t ranges over integers, t^2 mod p takes (p+1)/2 values.
///   So (D|p) is determined by t mod p and l mod p.
///
/// This function analyzes how the Kronecker symbols distribute and
/// whether their pattern reveals anything about p.
#[derive(Debug, Clone)]
pub struct KroneckerDistribution {
    pub n: u64,
    pub p: u64,
    pub q: u64,
    pub l: u64,
    /// Count of discriminants with (D|p) = 1.
    pub count_split_p: usize,
    /// Count of discriminants with (D|p) = -1.
    pub count_inert_p: usize,
    /// Count of discriminants with (D|p) = 0.
    pub count_ramified_p: usize,
    /// Same for q.
    pub count_split_q: usize,
    pub count_inert_q: usize,
    pub count_ramified_q: usize,
    /// Correlation coefficient between (D|p) and (D|q).
    pub kronecker_correlation: f64,
}

/// Compute the Kronecker symbol distribution for a specific (N, l) pair.
pub fn kronecker_distribution(n: u64, p: u64, q: u64, l: u64) -> KroneckerDistribution {
    let analysis = analyze_discriminants(n, p, q, l);

    let mut count_split_p = 0usize;
    let mut count_inert_p = 0usize;
    let mut count_ramified_p = 0usize;
    let mut count_split_q = 0usize;
    let mut count_inert_q = 0usize;
    let mut count_ramified_q = 0usize;

    let mut sum_kp = 0.0f64;
    let mut sum_kq = 0.0f64;
    let mut sum_kp_sq = 0.0f64;
    let mut sum_kq_sq = 0.0f64;
    let mut sum_kp_kq = 0.0f64;

    for info in &analysis.discriminants {
        match info.kronecker_p {
            1 => count_split_p += 1,
            -1 => count_inert_p += 1,
            0 => count_ramified_p += 1,
            _ => {}
        }
        match info.kronecker_q {
            1 => count_split_q += 1,
            -1 => count_inert_q += 1,
            0 => count_ramified_q += 1,
            _ => {}
        }

        let kp = info.kronecker_p as f64;
        let kq = info.kronecker_q as f64;
        sum_kp += kp;
        sum_kq += kq;
        sum_kp_sq += kp * kp;
        sum_kq_sq += kq * kq;
        sum_kp_kq += kp * kq;
    }

    let n_pts = analysis.discriminants.len() as f64;
    let kronecker_correlation = if n_pts > 1.0 {
        let mean_kp = sum_kp / n_pts;
        let mean_kq = sum_kq / n_pts;
        let var_kp = sum_kp_sq / n_pts - mean_kp * mean_kp;
        let var_kq = sum_kq_sq / n_pts - mean_kq * mean_kq;
        let cov = sum_kp_kq / n_pts - mean_kp * mean_kq;

        if var_kp > 1e-10 && var_kq > 1e-10 {
            cov / (var_kp.sqrt() * var_kq.sqrt())
        } else {
            0.0
        }
    } else {
        0.0
    };

    KroneckerDistribution {
        n,
        p,
        q,
        l,
        count_split_p,
        count_inert_p,
        count_ramified_p,
        count_split_q,
        count_inert_q,
        count_ramified_q,
        kronecker_correlation,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_analyze_discriminants_basic() {
        // N = 77 = 7 * 11, l = 2
        let analysis = analyze_discriminants(77, 7, 11, 2);
        assert_eq!(analysis.n, 77);
        assert_eq!(analysis.p, 7);
        assert_eq!(analysis.q, 11);
        assert!(!analysis.discriminants.is_empty());

        // Verify splitting pattern counts add up
        let sp = &analysis.splitting_pattern;
        assert_eq!(
            sp.split_both + sp.split_p_only + sp.split_q_only + sp.split_neither,
            sp.total
        );
    }

    #[test]
    fn test_analyze_discriminants_multi_l() {
        let analyses = analyze_discriminants_multi_l(77, 7, 11, &[2, 3, 5]);
        assert_eq!(analyses.len(), 3);
        for (i, &l) in [2u64, 3, 5].iter().enumerate() {
            assert_eq!(analyses[i].l, l);
        }
    }

    #[test]
    fn test_kronecker_multiplicativity_sanity() {
        // (D|N) = (D|p) * (D|q) for N = pq
        // This should always hold by the Kronecker symbol definition
        let result = check_class_number_multiplicativity(77, 7, 11);
        for dp in &result.data_points {
            assert_eq!(
                dp.kronecker_n,
                dp.kronecker_p * dp.kronecker_q,
                "Kronecker multiplicativity failed for D={}",
                dp.d
            );
        }
    }

    #[test]
    fn test_kronecker_distribution() {
        let dist = kronecker_distribution(77, 7, 11, 2);
        assert_eq!(dist.n, 77);
        // Total should match discriminant count
        let total = dist.count_split_p + dist.count_inert_p + dist.count_ramified_p;
        assert!(total > 0, "Should have at least one discriminant");
    }

    #[test]
    fn test_splitting_pattern_consistency() {
        // For N = 143 = 11 * 13, l = 3
        let analysis = analyze_discriminants(143, 11, 13, 3);
        let sp = &analysis.splitting_pattern;

        // By quadratic reciprocity, the number of QRs mod p among t^2 - 4l
        // should be roughly half for large enough samples
        assert!(sp.total > 0);

        // H sums should be non-negative
        assert!(sp.h_sum_split_both >= 0.0);
        assert!(sp.h_sum_split_p >= 0.0);
        assert!(sp.h_sum_split_q >= 0.0);
        assert!(sp.h_sum_split_neither >= 0.0);
    }

    #[test]
    fn test_multiplicativity_result_structure() {
        let result = check_class_number_multiplicativity(77, 7, 11);
        assert_eq!(result.n, 77);
        assert_eq!(result.p, 7);
        assert_eq!(result.q, 11);
        assert!(!result.data_points.is_empty());
        // Relative error should be non-negative
        assert!(result.avg_relative_error >= 0.0);
        assert!(result.max_relative_error >= 0.0);
    }
}
