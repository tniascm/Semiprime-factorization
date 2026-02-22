//! Lattice reduction for integer factorization.
//!
//! Implements LLL algorithm and explores Schnorr's SVP-based factoring.
//! Goal: independently verify Ducas' experimental findings.

use num_bigint::BigUint;
use num_traits::ToPrimitive;

/// A lattice basis represented as a matrix of rational vectors.
/// Each row is a basis vector.
pub type LatticeBasis = Vec<Vec<f64>>;

/// Result of a Ducas-style verification experiment.
///
/// Ducas showed empirically that Schnorr's lattice-based factoring approach
/// yields essentially 0 out of 1000 useful relations. This struct captures
/// the same kind of statistics so we can replicate that finding.
#[derive(Debug, Clone)]
pub struct DucasResult {
    /// Total number of factoring attempts
    pub num_trials: usize,
    /// How many attempts yielded a non-trivial factor
    pub num_successes: usize,
    /// `num_successes / num_trials`
    pub success_rate: f64,
    /// Mean Euclidean norm of the shortest vector across all reduced bases
    pub avg_shortest_vector_norm: f64,
}

/// Quality metrics for a lattice basis after reduction.
#[derive(Debug, Clone)]
pub struct BasisQuality {
    /// Hermite factor: ||b_1|| / det(L)^(1/n).
    /// Measures how short the first vector is relative to the lattice determinant.
    /// For a perfect lattice, this equals 1.
    pub hermite_factor: f64,
    /// Orthogonality defect: product(||b_i||) / det(L).
    /// Measures how far the basis vectors are from being mutually orthogonal.
    /// Equals 1 for an orthogonal basis.
    pub orthogonality_defect: f64,
    /// Euclidean norm of the shortest (first) basis vector.
    pub shortest_vector_norm: f64,
}

/// LLL-reduced basis parameters.
#[derive(Debug, Clone)]
pub struct LllParams {
    /// Lovász condition parameter (typically 0.75)
    pub delta: f64,
}

impl Default for LllParams {
    fn default() -> Self {
        Self { delta: 0.75 }
    }
}

/// Gram-Schmidt orthogonalization.
pub fn gram_schmidt(basis: &LatticeBasis) -> (LatticeBasis, Vec<Vec<f64>>) {
    let n = basis.len();
    let m = basis[0].len();
    let mut ortho = basis.clone();
    let mut mu = vec![vec![0.0; n]; n];

    for i in 0..n {
        for j in 0..i {
            let dot_ij: f64 = (0..m).map(|k| basis[i][k] * ortho[j][k]).sum();
            let dot_jj: f64 = (0..m).map(|k| ortho[j][k] * ortho[j][k]).sum();
            mu[i][j] = if dot_jj > 1e-10 { dot_ij / dot_jj } else { 0.0 };

            for k in 0..m {
                ortho[i][k] -= mu[i][j] * ortho[j][k];
            }
        }
    }

    (ortho, mu)
}

/// LLL lattice basis reduction algorithm.
pub fn lll_reduce(basis: &mut LatticeBasis, params: &LllParams) {
    let n = basis.len();
    if n == 0 {
        return;
    }
    let m = basis[0].len();

    let mut k = 1;
    while k < n {
        let (_ortho, mu) = gram_schmidt(basis);

        // Size reduction
        for j in (0..k).rev() {
            if mu[k][j].abs() > 0.5 {
                let r = mu[k][j].round();
                for l in 0..m {
                    basis[k][l] -= r * basis[j][l];
                }
            }
        }

        // Lovász condition check
        let (ortho, mu) = gram_schmidt(basis);
        let norm_k: f64 = (0..m).map(|l| ortho[k][l].powi(2)).sum();
        let norm_k1: f64 = (0..m).map(|l| ortho[k - 1][l].powi(2)).sum();

        if norm_k >= (params.delta - mu[k][k - 1].powi(2)) * norm_k1 {
            k += 1;
        } else {
            basis.swap(k, k - 1);
            k = k.max(1);
            if k > 1 {
                k -= 1;
            }
        }
    }
}

/// Construct Schnorr's factoring lattice for integer N.
/// The lattice is constructed to find smooth relations.
pub fn schnorr_factoring_lattice(n: &BigUint, _dimension: usize, primes: &[u64]) -> LatticeBasis {
    let k = primes.len();
    let dim = k + 1;

    // Scaling constant C: controls the trade-off between finding short
    // exponent vectors and satisfying the smooth-relation constraint.
    // A common choice is N^(1/k) but a simpler heuristic works for experiments.
    let ln_n = n.to_f64().unwrap_or(f64::MAX).ln();
    let c = (ln_n * (k as f64)).sqrt().ceil().max(1.0);

    // Scaling factor M for the logarithmic column so that rounding
    // errors stay small relative to the integer entries.
    let m = c * c;

    let mut basis = vec![vec![0.0; dim]; dim];

    // Rows 0..k: one row per factor-base prime p_i
    for i in 0..k {
        // Standard basis vector e_i scaled by C
        basis[i][i] = c;
        // Last column: round(M * ln(p_i))
        let ln_pi = (primes[i] as f64).ln();
        basis[i][k] = (m * ln_pi).round();
    }

    // Last row: encodes N
    // All entries zero except the last column: round(M * ln(N))
    basis[k][k] = (m * ln_n).round();

    basis
}

/// Sieve primes up to `bound` using trial division.
fn sieve_primes(bound: u64) -> Vec<u64> {
    if bound < 2 {
        return vec![];
    }
    let mut is_prime = vec![true; (bound + 1) as usize];
    is_prime[0] = false;
    if bound >= 1 {
        is_prime[1] = false;
    }
    let mut p = 2u64;
    while p * p <= bound {
        if is_prime[p as usize] {
            let mut multiple = p * p;
            while multiple <= bound {
                is_prime[multiple as usize] = false;
                multiple += p;
            }
        }
        p += 1;
    }
    (2..=bound).filter(|&i| is_prime[i as usize]).collect()
}

/// Check if a number is smooth with respect to the given factor base.
/// Returns the exponent vector if smooth, None otherwise.
#[allow(dead_code)]
fn smooth_factorize(mut val: BigUint, primes: &[u64]) -> Option<Vec<i64>> {
    let one = BigUint::from(1u32);
    let mut exponents = vec![0i64; primes.len()];

    for (i, &p) in primes.iter().enumerate() {
        let p_big = BigUint::from(p);
        while &val % &p_big == BigUint::from(0u32) {
            val /= &p_big;
            exponents[i] += 1;
        }
    }

    if val == one {
        Some(exponents)
    } else {
        None
    }
}

/// Attempt to factor N using lattice reduction (Schnorr's approach).
pub fn lattice_factor(n: &BigUint, dimension: usize) -> Option<BigUint> {
    use factoring_core::gcd;

    let one = BigUint::from(1u32);

    // 1. Sieve small primes for the factor base
    let prime_bound = (dimension as u64) * 10;
    let primes = sieve_primes(prime_bound);
    let primes: Vec<u64> = primes.into_iter().take(dimension.saturating_sub(1).max(2)).collect();

    // 2. Construct Schnorr's lattice
    let mut basis = schnorr_factoring_lattice(n, dimension, &primes);

    // 3. LLL-reduce it
    let params = LllParams::default();
    lll_reduce(&mut basis, &params);

    // 4. Extract smooth relations from short vectors
    let k = primes.len();
    for row in &basis {
        // The first k entries are the exponent vector (scaled by C),
        // recover integer exponents by rounding
        let ln_n = n.to_f64().unwrap_or(f64::MAX).ln();
        let c = (ln_n * (k as f64)).sqrt().ceil().max(1.0);
        let exponents: Vec<i64> = row[..k].iter().map(|&v| (v / c).round() as i64).collect();

        // Skip the zero vector
        if exponents.iter().all(|&e| e == 0) {
            continue;
        }

        // Compute product(p_i^|a_i|) for positive and negative exponents separately
        let mut lhs = BigUint::from(1u32);
        let mut rhs = BigUint::from(1u32);

        for (i, &exp) in exponents.iter().enumerate() {
            let p_big = BigUint::from(primes[i]);
            if exp > 0 {
                for _ in 0..exp {
                    lhs *= &p_big;
                }
            } else if exp < 0 {
                for _ in 0..(-exp) {
                    rhs *= &p_big;
                }
            }
        }

        // Try gcd(lhs - rhs, n) and gcd(lhs + rhs, n)
        if lhs > rhs {
            let diff = &lhs - &rhs;
            let g = gcd(&diff, n);
            if g > one && g < *n {
                return Some(g);
            }
        } else if rhs > lhs {
            let diff = &rhs - &lhs;
            let g = gcd(&diff, n);
            if g > one && g < *n {
                return Some(g);
            }
        }

        let sum = &lhs + &rhs;
        let g = gcd(&sum, n);
        if g > one && g < *n {
            return Some(g);
        }

        // Also try gcd(lhs, n) and gcd(rhs, n) directly
        let g = gcd(&lhs, n);
        if g > one && g < *n {
            return Some(g);
        }
        let g = gcd(&rhs, n);
        if g > one && g < *n {
            return Some(g);
        }
    }

    // 5. Try smooth factorization of the relations against n
    for row in &basis {
        let ln_n = n.to_f64().unwrap_or(f64::MAX).ln();
        let c = (ln_n * (k as f64)).sqrt().ceil().max(1.0);
        let exponents: Vec<i64> = row[..k].iter().map(|&v| (v / c).round() as i64).collect();

        if exponents.iter().all(|&e| e == 0) {
            continue;
        }

        // Build the product from positive exponents
        let mut product = BigUint::from(1u32);
        for (i, &exp) in exponents.iter().enumerate() {
            let p_big = BigUint::from(primes[i]);
            let abs_exp = exp.unsigned_abs();
            for _ in 0..abs_exp {
                product *= &p_big;
            }
        }

        let g = gcd(&product, n);
        if g > one && g < *n {
            return Some(g);
        }
    }

    None
}

/// Compute quality metrics for a lattice basis.
///
/// - **Hermite factor** = `||b_1|| / det(L)^(1/n)` where `n` is the dimension.
/// - **Orthogonality defect** = `product(||b_i||) / |det(L)|`.
/// - **Shortest vector norm** = `||b_1||` (the first vector after reduction).
///
/// The determinant is computed as the absolute value of the determinant of the
/// Gram-Schmidt orthogonalized basis (product of the norms of the orthogonal vectors).
pub fn basis_quality(basis: &LatticeBasis) -> BasisQuality {
    let n = basis.len();
    if n == 0 {
        return BasisQuality {
            hermite_factor: 0.0,
            orthogonality_defect: 0.0,
            shortest_vector_norm: 0.0,
        };
    }

    // Compute Gram-Schmidt to get orthogonal vectors (needed for determinant)
    let (ortho, _mu) = gram_schmidt(basis);

    // Norms of original basis vectors
    let norms: Vec<f64> = basis
        .iter()
        .map(|row| row.iter().map(|x| x * x).sum::<f64>().sqrt())
        .collect();

    // Norms of orthogonal basis vectors (used for determinant)
    let ortho_norms: Vec<f64> = ortho
        .iter()
        .map(|row| row.iter().map(|x| x * x).sum::<f64>().sqrt())
        .collect();

    let shortest_vector_norm = norms[0];

    // det(L) = product of orthogonal vector norms
    let log_det: f64 = ortho_norms.iter().map(|norm| norm.ln()).sum();
    let det_root_n = (log_det / n as f64).exp(); // det(L)^(1/n)

    let hermite_factor = if det_root_n > 1e-15 {
        shortest_vector_norm / det_root_n
    } else {
        f64::INFINITY
    };

    // Orthogonality defect = product(||b_i||) / det(L)
    let log_product_norms: f64 = norms.iter().map(|norm| norm.ln()).sum();
    let orthogonality_defect = (log_product_norms - log_det).exp();

    BasisQuality {
        hermite_factor,
        orthogonality_defect,
        shortest_vector_norm,
    }
}

/// Replicate Ducas' experimental verification of Schnorr's lattice factoring approach.
///
/// Runs `lattice_factor` on `n` across `num_trials` attempts, each with a
/// slightly varying dimension (from `dimension` down to `dimension / 2 + 1`),
/// cycling through them. Tracks:
///
/// - How many attempts yield a non-trivial factor (`num_successes`).
/// - The average norm of the shortest vector in each LLL-reduced basis.
///
/// Ducas' original experiment found a 0% success rate across 1000 trials,
/// demonstrating that Schnorr's approach does not produce useful relations.
pub fn ducas_verification(n: &BigUint, dimension: usize, num_trials: usize) -> DucasResult {
    let mut num_successes = 0usize;
    let mut total_shortest_norm = 0.0f64;

    let min_dim = (dimension / 2).max(2) + 1;

    for trial in 0..num_trials {
        // Vary dimension across trials to explore different lattice sizes
        let dim = min_dim + (trial % (dimension - min_dim + 1));

        // Build factor base and lattice
        let prime_bound = (dim as u64) * 10;
        let primes = sieve_primes(prime_bound);
        let fb: Vec<u64> = primes.into_iter().take(dim.saturating_sub(1).max(2)).collect();

        let mut basis = schnorr_factoring_lattice(n, dim, &fb);
        let params = LllParams::default();
        lll_reduce(&mut basis, &params);

        // Record shortest vector norm
        let first_norm: f64 = basis[0].iter().map(|x| x * x).sum::<f64>().sqrt();
        total_shortest_norm += first_norm;

        // Check if lattice_factor succeeds at this dimension
        if lattice_factor(n, dim).is_some() {
            num_successes += 1;
        }
    }

    let success_rate = if num_trials > 0 {
        num_successes as f64 / num_trials as f64
    } else {
        0.0
    };

    let avg_shortest_vector_norm = if num_trials > 0 {
        total_shortest_norm / num_trials as f64
    } else {
        0.0
    };

    DucasResult {
        num_trials,
        num_successes,
        success_rate,
        avg_shortest_vector_norm,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_schnorr_lattice_dimensions_and_lll_shortens() {
        let n = BigUint::from(1001u32); // 7 * 11 * 13
        let primes = vec![2, 3, 5, 7, 11];
        let dimension = primes.len() + 1;

        let basis = schnorr_factoring_lattice(&n, dimension, &primes);

        // Check dimensions: (k+1) x (k+1)
        assert_eq!(basis.len(), dimension);
        for row in &basis {
            assert_eq!(row.len(), dimension);
        }

        // Compute norms before LLL
        let norms_before: Vec<f64> = basis
            .iter()
            .map(|row| row.iter().map(|x| x.powi(2)).sum::<f64>().sqrt())
            .collect();

        // LLL-reduce and check that the first vector gets shorter
        let mut reduced = basis.clone();
        let params = LllParams::default();
        lll_reduce(&mut reduced, &params);

        // Dimensions should be preserved
        assert_eq!(reduced.len(), dimension);
        for row in &reduced {
            assert_eq!(row.len(), dimension);
        }

        let norms_after: Vec<f64> = reduced
            .iter()
            .map(|row| row.iter().map(|x| x.powi(2)).sum::<f64>().sqrt())
            .collect();

        // After LLL reduction, the first (shortest) vector should be no longer
        // than the shortest vector from the original basis
        let min_before = norms_before.iter().cloned().fold(f64::MAX, f64::min);
        assert!(
            norms_after[0] <= min_before + 1e-6,
            "LLL should produce a first vector at most as long as the shortest original: {} vs {}",
            norms_after[0],
            min_before
        );
    }

    #[test]
    fn test_ducas_verification() {
        // Small semiprime: 103 * 149 = 15347
        let n = BigUint::from(15347u32);
        let result = ducas_verification(&n, 8, 10);

        // Verify result struct is properly populated
        assert_eq!(result.num_trials, 10);
        assert!(result.num_successes <= result.num_trials);
        assert!(result.success_rate >= 0.0 && result.success_rate <= 1.0);
        assert!(
            result.avg_shortest_vector_norm > 0.0,
            "Average shortest vector norm should be positive, got {}",
            result.avg_shortest_vector_norm
        );

        // The success rate should be consistent with the counts
        let expected_rate = result.num_successes as f64 / result.num_trials as f64;
        assert!(
            (result.success_rate - expected_rate).abs() < 1e-10,
            "Success rate {} doesn't match num_successes/num_trials = {}",
            result.success_rate,
            expected_rate
        );
    }

    #[test]
    fn test_basis_quality() {
        // Identity matrix: a trivially orthogonal basis
        let identity: LatticeBasis = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];

        let quality = basis_quality(&identity);

        // For the identity matrix:
        // - Each vector has norm 1, so shortest_vector_norm = 1
        // - det(L) = 1, so det(L)^(1/3) = 1
        // - hermite_factor = 1/1 = 1
        // - product(||b_i||) = 1*1*1 = 1, det(L) = 1, so orthogonality_defect = 1
        assert!(
            (quality.hermite_factor - 1.0).abs() < 1e-10,
            "Identity basis should have hermite_factor = 1.0, got {}",
            quality.hermite_factor
        );
        assert!(
            (quality.orthogonality_defect - 1.0).abs() < 1e-10,
            "Identity basis should have orthogonality_defect = 1.0, got {}",
            quality.orthogonality_defect
        );
        assert!(
            (quality.shortest_vector_norm - 1.0).abs() < 1e-10,
            "Identity basis should have shortest_vector_norm = 1.0, got {}",
            quality.shortest_vector_norm
        );

        // A non-orthogonal basis should have orthogonality_defect > 1
        let skewed: LatticeBasis = vec![
            vec![1.0, 1.0, 0.0],
            vec![0.0, 1.0, 1.0],
            vec![1.0, 0.0, 1.0],
        ];

        let quality_skewed = basis_quality(&skewed);
        assert!(
            quality_skewed.orthogonality_defect >= 1.0 - 1e-10,
            "Non-orthogonal basis should have orthogonality_defect >= 1.0, got {}",
            quality_skewed.orthogonality_defect
        );
    }

    #[test]
    fn test_basis_quality_after_lll() {
        // Verify that LLL reduction improves basis quality
        let mut basis: LatticeBasis = vec![
            vec![1.0, 1.0, 1.0],
            vec![-1.0, 0.0, 2.0],
            vec![3.0, 5.0, 6.0],
        ];

        let quality_before = basis_quality(&basis);

        let params = LllParams::default();
        lll_reduce(&mut basis, &params);

        let quality_after = basis_quality(&basis);

        // After LLL, orthogonality defect should not increase significantly
        // and the shortest vector should be no longer
        assert!(
            quality_after.shortest_vector_norm <= quality_before.shortest_vector_norm + 1e-6,
            "LLL should not increase shortest vector norm: before={}, after={}",
            quality_before.shortest_vector_norm,
            quality_after.shortest_vector_norm
        );
    }

    #[test]
    fn test_lll_simple() {
        let mut basis = vec![
            vec![1.0, 1.0, 1.0],
            vec![-1.0, 0.0, 2.0],
            vec![3.0, 5.0, 6.0],
        ];
        let params = LllParams::default();
        lll_reduce(&mut basis, &params);

        // After LLL, vectors should be shorter
        let norm0: f64 = basis[0].iter().map(|x| x.powi(2)).sum();
        assert!(norm0 < 20.0); // Reduced basis should have reasonable norms
    }
}
