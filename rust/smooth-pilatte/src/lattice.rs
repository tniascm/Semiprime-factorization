//! Pilatte lattice construction for smooth relation finding.
//!
//! Based on Pilatte (2024) zero-density estimates: elements of (Z/NZ)* can be
//! written as short products of the first d small primes where d = O(sqrt(n))
//! and exponents are bounded by O(log n). This module constructs lattices whose
//! short vectors encode smooth relations, then reduces them with LLL.

use lattice_reduction::{basis_quality, lll_reduce, BasisQuality, LatticeBasis, LllParams};
use num_bigint::BigUint;
use num_traits::ToPrimitive;

/// Parameters for the Pilatte lattice construction.
#[derive(Debug, Clone)]
pub struct PilatteParams {
    /// Number of bits in N.
    pub bit_size: u32,
    /// Factor base dimension d (number of small primes).
    pub dimension: usize,
    /// The factor base primes.
    pub primes: Vec<u64>,
    /// Scaling constant C = sqrt(ln(N) * d).
    pub c_scale: f64,
    /// Exponent bound from Pilatte's theorem: O(log n).
    pub exponent_bound: f64,
    /// Smoothness probability estimate (Canfield-Erdos-Pomerance).
    pub smooth_probability: f64,
}

/// Result of building and reducing a Pilatte lattice.
#[derive(Debug, Clone)]
pub struct PilatteLatticeResult {
    /// The LLL-reduced basis.
    pub basis: LatticeBasis,
    /// Parameters used for construction.
    pub params: PilatteParams,
    /// Quality metrics of the reduced basis.
    pub quality: BasisQuality,
}

/// Sieve primes up to bound.
pub fn sieve_primes(bound: u64) -> Vec<u64> {
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

/// Compute the first `count` primes.
pub fn first_n_primes(count: usize) -> Vec<u64> {
    if count == 0 {
        return vec![];
    }
    // Upper bound for the n-th prime: p_n < n * (ln(n) + ln(ln(n))) for n >= 6
    let bound = if count < 6 {
        20u64
    } else {
        let n = count as f64;
        (n * (n.ln() + n.ln().ln()) * 1.3) as u64
    };
    let all = sieve_primes(bound);
    all.into_iter().take(count).collect()
}

/// Compute Pilatte dimension d from bit size n.
///
/// Pilatte's theorem: d = O(sqrt(n)). We use d = ceil(sqrt(n))
/// as baseline following the proof's construction.
pub fn pilatte_dimension(bit_size: u32) -> usize {
    let n = bit_size as f64;
    let d = n.sqrt().ceil() as usize;
    d.max(4) // minimum 4 primes for meaningful lattice
}

/// Compute smoothness probability using Canfield-Erdos-Pomerance.
///
/// The probability that a random number <= x is y-smooth:
///   Psi(x, y) / x ≈ u^{-u} where u = log(x) / log(y)
fn smooth_probability(n_bits: u32, smoothness_bound: u64) -> f64 {
    let log_x = (n_bits as f64) * std::f64::consts::LN_2;
    let log_y = (smoothness_bound as f64).ln();
    if log_y <= 0.0 {
        return 0.0;
    }
    let u = log_x / log_y;
    if u <= 0.0 {
        return 1.0;
    }
    // Dickman rho approximation: rho(u) ≈ u^{-u} for large u
    (-u * u.ln()).exp()
}

/// Build the Pilatte lattice for integer N.
///
/// The lattice encodes the constraint: find exponent vectors e = (e_1, ..., e_d)
/// such that p_1^{e_1} * ... * p_d^{e_d} ≡ something smooth (mod N).
///
/// Construction (d+1) x (d+1):
///   - Rows 0..d-1: standard basis e_i scaled by C, plus M*ln(p_i) in last column
///   - Row d: zeros except M*ln(N) in last column
///
/// The LLL-reduced short vectors have small exponent entries (bounded by O(log n))
/// and encode smooth relations modulo N.
pub fn build_pilatte_lattice(n: &BigUint, dimension: usize) -> PilatteLatticeResult {
    let primes = first_n_primes(dimension);
    build_pilatte_lattice_with_primes(n, &primes)
}

/// Build the Pilatte lattice with explicit factor base.
pub fn build_pilatte_lattice_with_primes(n: &BigUint, primes: &[u64]) -> PilatteLatticeResult {
    let d = primes.len();
    let dim = d + 1;
    let n_bits = n.bits() as u32;

    let ln_n = n.to_f64().unwrap_or(f64::MAX).ln();

    // Pilatte scaling: C = sqrt(ln(N) * d)
    // Balances exponent vector length against the log-constraint residual
    let c = (ln_n * (d as f64)).sqrt().ceil().max(1.0);

    // M = C^2 for log-column scaling (ensures integer rounding stays precise)
    let m = c * c;

    // Exponent bound from Pilatte's theorem: |e_i| <= O(log n)
    let exponent_bound = (n_bits as f64).ln() * 2.0;

    // Build basis matrix
    let mut basis = vec![vec![0.0; dim]; dim];

    for i in 0..d {
        // Standard basis scaled by C
        basis[i][i] = c;
        // Log-constraint column
        let ln_pi = (primes[i] as f64).ln();
        basis[i][d] = (m * ln_pi).round();
    }

    // Last row: encodes N (the modular constraint)
    basis[d][d] = (m * ln_n).round();

    // LLL-reduce
    let params = LllParams::default();
    lll_reduce(&mut basis, &params);

    let quality = basis_quality(&basis);
    let smooth_prob = smooth_probability(n_bits, *primes.last().unwrap_or(&2));

    let pilatte_params = PilatteParams {
        bit_size: n_bits,
        dimension: d,
        primes: primes.to_vec(),
        c_scale: c,
        exponent_bound,
        smooth_probability: smooth_prob,
    };

    PilatteLatticeResult {
        basis,
        params: pilatte_params,
        quality,
    }
}

/// Build a weighted Pilatte lattice that emphasizes smaller primes.
///
/// Pilatte's zero-density proof implies that smooth representations favor
/// smaller primes. We weight the lattice rows to make LLL prefer short vectors
/// with larger exponents on small primes and smaller exponents on large primes.
pub fn build_weighted_pilatte_lattice(n: &BigUint, dimension: usize) -> PilatteLatticeResult {
    let primes = first_n_primes(dimension);
    let d = primes.len();
    let dim = d + 1;
    let n_bits = n.bits() as u32;

    let ln_n = n.to_f64().unwrap_or(f64::MAX).ln();
    let c_base = (ln_n * (d as f64)).sqrt().ceil().max(1.0);
    let m = c_base * c_base;

    let exponent_bound = (n_bits as f64).ln() * 2.0;

    let mut basis = vec![vec![0.0; dim]; dim];

    let ln_p0 = (primes[0] as f64).ln().max(0.5);
    for i in 0..d {
        // Weight inversely proportional to log(p_i):
        // Smaller primes get smaller scaling -> LLL allows larger exponents for them
        let weight = (primes[i] as f64).ln().max(0.5);
        let c_i = c_base * weight / ln_p0;
        basis[i][i] = c_i;
        let ln_pi = (primes[i] as f64).ln();
        basis[i][d] = (m * ln_pi).round();
    }

    basis[d][d] = (m * ln_n).round();

    let params = LllParams::default();
    lll_reduce(&mut basis, &params);

    let quality = basis_quality(&basis);
    let smooth_prob = smooth_probability(n_bits, *primes.last().unwrap_or(&2));

    let pilatte_params = PilatteParams {
        bit_size: n_bits,
        dimension: d,
        primes: primes.to_vec(),
        c_scale: c_base,
        exponent_bound,
        smooth_probability: smooth_prob,
    };

    PilatteLatticeResult {
        basis,
        params: pilatte_params,
        quality,
    }
}

/// Extract integer exponent vectors from the reduced basis.
///
/// Recovers the exponent vector (e_1, ..., e_d) from each row by dividing
/// by the scaling constant C and rounding. Skips rows whose exponents are
/// all zero or exceed the Pilatte bound.
pub fn extract_exponent_vectors(result: &PilatteLatticeResult) -> Vec<Vec<i64>> {
    let d = result.params.dimension;
    let c = result.params.c_scale;
    let bound = result.params.exponent_bound;
    let mut vectors = Vec::new();

    for row in &result.basis {
        let exponents: Vec<i64> = row[..d].iter().map(|&v| (v / c).round() as i64).collect();

        // Skip zero vector
        if exponents.iter().all(|&e| e == 0) {
            continue;
        }

        // Skip vectors with exponents exceeding Pilatte's bound (with slack)
        let max_exp = exponents.iter().map(|e| e.unsigned_abs()).max().unwrap_or(0);
        if max_exp as f64 > bound * 3.0 {
            continue;
        }

        vectors.push(exponents);
    }

    vectors
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_first_n_primes() {
        let primes = first_n_primes(10);
        assert_eq!(primes, vec![2, 3, 5, 7, 11, 13, 17, 19, 23, 29]);
    }

    #[test]
    fn test_pilatte_dimension() {
        assert_eq!(pilatte_dimension(16), 4);
        assert_eq!(pilatte_dimension(64), 8);
        assert_eq!(pilatte_dimension(128), 12);
        assert_eq!(pilatte_dimension(256), 16);
    }

    #[test]
    fn test_build_pilatte_lattice() {
        let n = BigUint::from(15347u64); // 103 * 149
        let result = build_pilatte_lattice(&n, 6);

        // Check dimensions: (d+1) x (d+1) = 7x7
        assert_eq!(result.basis.len(), 7);
        for row in &result.basis {
            assert_eq!(row.len(), 7);
        }

        assert!(result.quality.shortest_vector_norm > 0.0);
        assert!(result.quality.hermite_factor > 0.0);
        assert_eq!(result.params.dimension, 6);
        assert_eq!(result.params.primes.len(), 6);
    }

    #[test]
    fn test_extract_exponent_vectors() {
        let n = BigUint::from(15347u64);
        let result = build_pilatte_lattice(&n, 6);
        let vectors = extract_exponent_vectors(&result);

        assert!(!vectors.is_empty(), "Should extract at least one exponent vector");
        for v in &vectors {
            assert_eq!(v.len(), 6);
        }
    }

    #[test]
    fn test_weighted_vs_standard() {
        let n = BigUint::from(15347u64);
        let standard = build_pilatte_lattice(&n, 6);
        let weighted = build_weighted_pilatte_lattice(&n, 6);

        assert_eq!(standard.basis.len(), weighted.basis.len());
        assert!(weighted.quality.shortest_vector_norm > 0.0);
    }

    #[test]
    fn test_smooth_probability_decreases() {
        let prob_16 = smooth_probability(16, 29);
        let prob_64 = smooth_probability(64, 29);
        let prob_128 = smooth_probability(128, 29);

        assert!(prob_16 > prob_64);
        assert!(prob_64 > prob_128);
        assert!(prob_16 > 0.0);
    }
}
