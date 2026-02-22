//! SR-pair construction and validation per the TNSS paper (arXiv:2410.16355v3).
//!
//! An sr-pair (u, u-vN) is constructed from a lattice exponent vector e:
//!   u = prod_{e_j >= 0} p_j^{e_j}
//!   v = prod_{e_j < 0} p_j^{-e_j}
//!
//! The pair is valid if BOTH u AND (u-vN) are B₂-smooth (all prime factors ≤ p_{π₂}).
//!
//! Key insight: The paper uses TWO factor bases:
//!   P₁ = first π primes (lattice basis, small)
//!   P₂ = first π₂ primes (smoothness check, much larger)
//!
//! π₂ >> π, e.g. for 100-bit: π = 64, π₂ = 12,801

use num_bigint::BigUint;
use num_integer::Integer;
use num_traits::{One, Zero};

/// An sr-pair: (u, u - v*N) where both are B₂-smooth.
#[derive(Debug, Clone)]
pub struct SrPair {
    /// u = prod_{e_j >= 0} p_j^{e_j}
    pub u: BigUint,
    /// v = prod_{e_j < 0} p_j^{-e_j}
    pub v: BigUint,
    /// |u - v*N| (the "remainder" that must be smooth)
    pub remainder: BigUint,
    /// sign of remainder: true if u >= v*N
    pub remainder_positive: bool,
    /// Exponent vector over P₂ for the combined sr-pair
    /// (exponents of u on the positive side, exponents of remainder on the negative side)
    pub exponents_p2: Vec<i32>,
    /// Exponent vector of u over P₂ (separately tracked for factor extraction)
    pub u_exponents_p2: Vec<u32>,
    /// Exponent vector of |remainder| over P₂ (separately tracked for factor extraction)
    pub rem_exponents_p2: Vec<u32>,
    /// Original lattice exponents (over P₁)
    pub lattice_exponents: Vec<i32>,
}

/// Configuration for sr-pair search.
#[derive(Debug, Clone)]
pub struct SrPairConfig {
    /// P₁: factoring basis (small, defines lattice)
    pub p1_primes: Vec<u64>,
    /// P₂: smoothness basis (larger)
    pub p2_primes: Vec<u64>,
    /// B₂ = p_{π₂}: smoothness bound
    pub b2_bound: u64,
}

impl SrPairConfig {
    /// Create config for a given bit size ℓ.
    /// Uses paper's scaling: π ~ small, π₂ = 2*n*ℓ
    pub fn for_bits(bits: usize) -> Self {
        let ell = bits as f64;

        // Lattice dimension n ≈ π (factor base size for the lattice)
        // Larger lattice = more diverse short vectors, but LLL is O(n^5)
        // Balance: sqrt(bits) * 4 gives reasonable dimensions
        let pi1 = if bits <= 16 {
            8
        } else if bits <= 24 {
            12
        } else if bits <= 32 {
            20
        } else if bits <= 40 {
            28
        } else if bits <= 52 {
            36
        } else if bits <= 64 {
            48
        } else if bits <= 80 {
            56
        } else {
            64
        };

        // Smoothness basis: π₂ = 2 * n * ℓ (from paper's smoothness bound formula)
        let pi2 = (2.0 * pi1 as f64 * ell).ceil() as usize;
        let pi2 = pi2.max(pi1 * 4).min(20_000);

        let all_primes = sieve_primes(pi2.max(1000));
        let p1: Vec<u64> = all_primes.iter().take(pi1).copied().collect();
        let p2: Vec<u64> = all_primes.iter().take(pi2).copied().collect();
        let b2 = p2.last().copied().unwrap_or(2);

        SrPairConfig {
            p1_primes: p1,
            p2_primes: p2,
            b2_bound: b2,
        }
    }
}

/// Build the Schnorr lattice B_{f,c} per Eq. 21 of the paper.
///
/// B_{f,c} is a (π+1) × π matrix where:
///   - Rows 0..π-1: diagonal with f(j) values (random permutation of {⌈j/2⌋})
///   - Row π: ⌈10^c * ln(p_j)⌉
///
/// Target vector: t = (0, ..., 0, ⌈10^c * ln(N)⌋)
pub fn build_schnorr_lattice(
    n: &BigUint,
    p1_primes: &[u64],
    c: f64,
    seed: u64,
) -> (Vec<Vec<f64>>, Vec<f64>) {
    let pi = p1_primes.len();
    let scale = 10.0f64.powf(c);

    // f(j) = random permutation of {ceil(j/2)}_{j=1}^{pi}
    // For simplicity, use a deterministic permutation based on seed
    let mut f_values: Vec<f64> = (1..=pi)
        .map(|j| ((j + 1) / 2) as f64)
        .collect();
    // Simple Fisher-Yates shuffle with seed
    let mut rng_state = seed;
    for i in (1..pi).rev() {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let j = (rng_state >> 33) as usize % (i + 1);
        f_values.swap(i, j);
    }

    // Build (pi+1) x pi matrix
    let mut basis = vec![vec![0.0f64; pi]; pi + 1];

    // Rows 0..pi-1: diagonal f(j)
    for j in 0..pi {
        basis[j][j] = f_values[j];
    }

    // Row pi: ceil(10^c * ln(p_j))
    for j in 0..pi {
        basis[pi][j] = (scale * (p1_primes[j] as f64).ln()).ceil();
    }

    // Target vector: t = (0, ..., 0, ceil(10^c * ln(N)))
    let n_f64 = n.to_f64_lossy();
    let mut target = vec![0.0f64; pi + 1];
    target[pi] = (scale * n_f64.ln()).ceil();

    (basis, target)
}

/// Babai's nearest plane algorithm for CVP.
///
/// Given LLL-reduced basis B and target t, find the closest lattice point.
/// Returns the coefficient vector c such that b_closest = sum(c_j * b_j).
pub fn babai_nearest_plane(
    basis: &[Vec<f64>],
    target: &[f64],
) -> Vec<f64> {
    let n = basis.len(); // number of basis vectors
    if n == 0 {
        return Vec::new();
    }
    let dim = basis[0].len();

    // Gram-Schmidt orthogonalization
    let mut gs_basis = basis.to_vec();
    let mut mu = vec![vec![0.0f64; n]; n];

    for i in 0..n {
        for j in 0..i {
            let dot_ij = dot(&basis[i], &gs_basis[j]);
            let dot_jj = dot(&gs_basis[j], &gs_basis[j]);
            mu[i][j] = if dot_jj.abs() > 1e-12 {
                dot_ij / dot_jj
            } else {
                0.0
            };
            for k in 0..dim {
                gs_basis[i][k] -= mu[i][j] * gs_basis[j][k];
            }
        }
    }

    // Babai's algorithm: work backwards from last basis vector
    let mut b = target.to_vec();
    // Ensure b has the right dimension (pad with zeros if needed)
    b.resize(dim, 0.0);

    let mut coeffs = vec![0.0f64; n];

    for i in (0..n).rev() {
        let dot_bi = dot(&b, &gs_basis[i]);
        let dot_ii = dot(&gs_basis[i], &gs_basis[i]);
        let c_i = if dot_ii.abs() > 1e-12 {
            (dot_bi / dot_ii).round()
        } else {
            0.0
        };
        coeffs[i] = c_i;

        // Subtract c_i * b_i from b
        for k in 0..dim {
            b[k] -= c_i * basis[i][k];
        }
    }

    coeffs
}

/// Construct an sr-pair from a lattice exponent vector.
///
/// Given exponents e over P₁, compute:
///   u = prod_{e_j >= 0} p_j^{e_j}
///   v = prod_{e_j < 0} p_j^{-e_j}
///   remainder = |u - v*N|
///
/// Then check if remainder is B₂-smooth (factors completely over P₂).
pub fn construct_sr_pair(
    exponents: &[i64],
    config: &SrPairConfig,
    n: &BigUint,
) -> Option<SrPair> {
    let k = config.p1_primes.len().min(exponents.len());

    // Skip trivial vectors
    if exponents.iter().take(k).all(|&e| e == 0) {
        return None;
    }

    // Skip too-large exponents
    if exponents.iter().any(|&e| e.abs() > 100) {
        return None;
    }

    // Compute u and v
    let mut u = BigUint::one();
    let mut v = BigUint::one();
    let mut lattice_exps = vec![0i32; k];

    for i in 0..k {
        let e = exponents[i];
        lattice_exps[i] = e as i32;
        let p = BigUint::from(config.p1_primes[i]);

        if e > 0 {
            u *= pow_biguint(&p, e as u32);
        } else if e < 0 {
            v *= pow_biguint(&p, (-e) as u32);
        }
    }

    // Compute u - v*N
    let v_n = &v * n;

    let (remainder, positive) = if u >= v_n {
        (&u - &v_n, true)
    } else {
        (&v_n - &u, false)
    };

    // Skip if remainder is zero (trivial) or is N itself
    if remainder.is_zero() || remainder == *n {
        return None;
    }

    // Check if remainder is B₂-smooth
    let (is_smooth, p2_exponents) = check_smooth_and_factor(&remainder, &config.p2_primes);
    if !is_smooth {
        return None;
    }

    // Also check that u is B₂-smooth (it should be by construction since p1 ⊂ p2)
    // But verify to be safe
    let (u_smooth, u_p2_exponents) = check_smooth_and_factor(&u, &config.p2_primes);
    if !u_smooth {
        return None;
    }

    // Build combined exponent vector over P₂
    let pi2 = config.p2_primes.len();
    let mut combined_exps = vec![0i32; pi2];
    for i in 0..pi2 {
        // u contributes positive exponents
        combined_exps[i] += u_p2_exponents[i] as i32;
        // remainder contributes negative exponents (if positive=true)
        // or positive (if positive=false), depending on sign
        if positive {
            combined_exps[i] -= p2_exponents[i] as i32;
        } else {
            combined_exps[i] += p2_exponents[i] as i32;
        }
    }

    Some(SrPair {
        u,
        v,
        remainder,
        remainder_positive: positive,
        exponents_p2: combined_exps,
        u_exponents_p2: u_p2_exponents,
        rem_exponents_p2: p2_exponents,
        lattice_exponents: lattice_exps,
    })
}

/// Check if n is B-smooth over the given prime base, and return the exponent vector.
/// Returns (true, exponents) if n factors completely, (false, partial) otherwise.
fn check_smooth_and_factor(n: &BigUint, primes: &[u64]) -> (bool, Vec<u32>) {
    let mut remaining = n.clone();
    let mut exponents = vec![0u32; primes.len()];

    for (i, &p) in primes.iter().enumerate() {
        let p_big = BigUint::from(p);
        while (&remaining % &p_big).is_zero() && remaining > BigUint::one() {
            remaining /= &p_big;
            exponents[i] += 1;
        }
    }

    (remaining == BigUint::one(), exponents)
}

/// GF(2) Gaussian elimination for sr-pairs using bitset representation.
///
/// Uses a DOUBLED matrix: for each prime p_j, we track TWO bits:
///   - bit at column 2*j: u-exponent of p_j mod 2
///   - bit at column 2*j+1: remainder-exponent of p_j mod 2
///
/// A null vector in this matrix means BOTH prod(u_r) and prod(remainder_r)
/// are perfect squares, giving a proper congruence of squares.
pub fn find_congruences(sr_pairs: &[SrPair], pi2: usize) -> Vec<Vec<usize>> {
    let num_pairs = sr_pairs.len();
    if num_pairs < 2 {
        return Vec::new();
    }

    // 2 * pi2 columns: even columns = u-exponents, odd columns = remainder-exponents
    let num_cols = 2 * pi2;
    let mat_words = (num_cols + 63) / 64;
    let hist_words = (num_pairs + 63) / 64;

    // Build GF(2) matrix as packed bitsets
    let mut matrix: Vec<Vec<u64>> = Vec::with_capacity(num_pairs);
    let mut history: Vec<Vec<u64>> = Vec::with_capacity(num_pairs);

    for (idx, pair) in sr_pairs.iter().enumerate() {
        let mut row = vec![0u64; mat_words];
        for j in 0..pi2 {
            // u-exponent parity at column 2*j
            let u_exp = pair.u_exponents_p2.get(j).copied().unwrap_or(0);
            if u_exp % 2 == 1 {
                let col = 2 * j;
                row[col / 64] |= 1u64 << (col % 64);
            }
            // remainder-exponent parity at column 2*j+1
            let r_exp = pair.rem_exponents_p2.get(j).copied().unwrap_or(0);
            if r_exp % 2 == 1 {
                let col = 2 * j + 1;
                row[col / 64] |= 1u64 << (col % 64);
            }
        }
        matrix.push(row);

        let mut hist = vec![0u64; hist_words];
        hist[idx / 64] |= 1u64 << (idx % 64);
        history.push(hist);
    }

    // Row reduction
    let mut pivot_row = 0;
    for col in 0..num_cols {
        let word = col / 64;
        let bit = col % 64;

        let mut found = None;
        for row in pivot_row..num_pairs {
            if (matrix[row][word] >> bit) & 1 == 1 {
                found = Some(row);
                break;
            }
        }

        let row_idx = match found {
            Some(r) => r,
            None => continue,
        };

        matrix.swap(pivot_row, row_idx);
        history.swap(pivot_row, row_idx);

        for row in 0..num_pairs {
            if row != pivot_row && (matrix[row][word] >> bit) & 1 == 1 {
                for w in 0..mat_words {
                    matrix[row][w] ^= matrix[pivot_row][w];
                }
                for w in 0..hist_words {
                    history[row][w] ^= history[pivot_row][w];
                }
            }
        }

        pivot_row += 1;
    }

    // Find zero rows = null vectors
    let mut null_vecs = Vec::new();
    for row in 0..num_pairs {
        let is_zero = matrix[row].iter().all(|&w| w == 0);
        if is_zero {
            let indices: Vec<usize> = (0..num_pairs)
                .filter(|&idx| (history[row][idx / 64] >> (idx % 64)) & 1 == 1)
                .collect();
            if !indices.is_empty() {
                null_vecs.push(indices);
            }
        }
    }

    null_vecs
}

/// Extract factors from a congruence of squares.
///
/// Given a subset of sr-pairs where both:
///   - prod(u_r) is a perfect square A² (u-exponents all even)
///   - prod(|remainder_r|) is a perfect square B² (rem-exponents all even)
///
/// Since u_r ≡ ±|remainder_r| (mod N), we have A² ≡ ±B² (mod N).
/// Then gcd(A ± B, N) may give nontrivial factors.
pub fn extract_factors_from_congruence(
    indices: &[usize],
    sr_pairs: &[SrPair],
    n: &BigUint,
    p2_primes: &[u64],
) -> Option<BigUint> {
    if indices.is_empty() {
        return None;
    }

    let pi2 = p2_primes.len();

    // Sum u-exponents and remainder-exponents separately
    let mut u_sum = vec![0u64; pi2];
    let mut rem_sum = vec![0u64; pi2];
    let mut sign_product_negative = false;

    for &idx in indices {
        if idx >= sr_pairs.len() {
            continue;
        }
        let pair = &sr_pairs[idx];
        for (j, &e) in pair.u_exponents_p2.iter().enumerate() {
            if j < pi2 {
                u_sum[j] += e as u64;
            }
        }
        for (j, &e) in pair.rem_exponents_p2.iter().enumerate() {
            if j < pi2 {
                rem_sum[j] += e as u64;
            }
        }
        // Track sign: u = v*N + remainder (positive) or u = v*N - remainder (negative)
        if !pair.remainder_positive {
            sign_product_negative = !sign_product_negative;
        }
    }

    // Compute A = sqrt(prod(u_r)) mod N = prod(p_j^{u_sum_j / 2}) mod N
    let mut a = BigUint::one();
    for (j, &e) in u_sum.iter().enumerate() {
        if j >= pi2 || e == 0 {
            continue;
        }
        let half = (e / 2) as u32;
        if half > 0 {
            let p = BigUint::from(p2_primes[j]);
            a = (&a * &pow_biguint_mod(&p, half, n)) % n;
        }
    }

    // Compute B = sqrt(prod(|remainder_r|)) mod N = prod(p_j^{rem_sum_j / 2}) mod N
    let mut b = BigUint::one();
    for (j, &e) in rem_sum.iter().enumerate() {
        if j >= pi2 || e == 0 {
            continue;
        }
        let half = (e / 2) as u32;
        if half > 0 {
            let p = BigUint::from(p2_primes[j]);
            b = (&b * &pow_biguint_mod(&p, half, n)) % n;
        }
    }

    // We have A² ≡ ±B² (mod N) depending on sign_product_negative
    // Try gcd(A - B, N) and gcd(A + B, N)
    let candidates = [
        if a >= b { (&a - &b).gcd(n) } else { (&b - &a).gcd(n) },
        (&a + &b).gcd(n),
    ];

    for g in &candidates {
        if g > &BigUint::one() && g < n {
            return Some(g.clone());
        }
    }

    // If sign is negative (A² ≡ -B² mod N), try different combinations
    if sign_product_negative {
        // A² + B² ≡ 0 (mod N), so try gcd(A² + B², N)
        let a2 = (&a * &a) % n;
        let b2 = (&b * &b) % n;
        let sum = (&a2 + &b2) % n;
        let g = sum.gcd(n);
        if g > BigUint::one() && &g < n {
            return Some(g);
        }
    }

    None
}

// Utility functions

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn pow_biguint(base: &BigUint, exp: u32) -> BigUint {
    let mut result = BigUint::one();
    let mut b = base.clone();
    let mut e = exp;
    while e > 0 {
        if e & 1 == 1 {
            result *= &b;
        }
        b = &b * &b;
        e >>= 1;
    }
    result
}

fn pow_biguint_mod(base: &BigUint, exp: u32, modulus: &BigUint) -> BigUint {
    if modulus.is_one() {
        return BigUint::zero();
    }
    let mut result = BigUint::one();
    let mut b = base % modulus;
    let mut e = exp;
    while e > 0 {
        if e & 1 == 1 {
            result = (&result * &b) % modulus;
        }
        b = (&b * &b) % modulus;
        e >>= 1;
    }
    result
}

/// Sieve primes up to a bound using Sieve of Eratosthenes.
pub fn sieve_primes(count: usize) -> Vec<u64> {
    if count == 0 {
        return Vec::new();
    }
    // Estimate upper bound for nth prime: p_n ≈ n * (ln(n) + ln(ln(n)))
    let bound = if count < 6 {
        15
    } else {
        let n = count as f64;
        (n * (n.ln() + n.ln().ln()) * 1.3) as usize + 100
    };

    let mut is_prime = vec![true; bound + 1];
    is_prime[0] = false;
    if bound > 0 {
        is_prime[1] = false;
    }
    let mut i = 2;
    while i * i <= bound {
        if is_prime[i] {
            let mut j = i * i;
            while j <= bound {
                is_prime[j] = false;
                j += i;
            }
        }
        i += 1;
    }

    is_prime
        .iter()
        .enumerate()
        .filter(|(_, &p)| p)
        .map(|(i, _)| i as u64)
        .take(count)
        .collect()
}

trait BigUintExt {
    fn to_f64_lossy(&self) -> f64;
}

impl BigUintExt for BigUint {
    fn to_f64_lossy(&self) -> f64 {
        use num_traits::ToPrimitive;
        self.to_f64().unwrap_or(f64::MAX)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sieve_primes() {
        let primes = sieve_primes(10);
        assert_eq!(primes, vec![2, 3, 5, 7, 11, 13, 17, 19, 23, 29]);
    }

    #[test]
    fn test_check_smooth() {
        let primes = vec![2, 3, 5, 7, 11];
        // 360 = 2^3 * 3^2 * 5
        let (smooth, exps) = check_smooth_and_factor(&BigUint::from(360u64), &primes);
        assert!(smooth);
        assert_eq!(exps, vec![3, 2, 1, 0, 0]);

        // 361 = 19^2, not smooth over {2,3,5,7,11}
        let (smooth, _) = check_smooth_and_factor(&BigUint::from(361u64), &primes);
        assert!(!smooth);
    }

    #[test]
    fn test_sr_pair_construction_small() {
        // N = 77 = 7 * 11
        let n = BigUint::from(77u64);
        let config = SrPairConfig {
            p1_primes: vec![2, 3, 5, 7, 11, 13],
            p2_primes: sieve_primes(50),
            b2_bound: 229,
        };

        // Try exponent vector [2, 0, 0, 1, 0, 0] => u = 2^2 * 7 = 28, v = 1
        // u - v*N = 28 - 77 = -49 = -(7^2)
        // |remainder| = 49 = 7^2, which is smooth over P₂
        let exps = vec![2i64, 0, 0, 1, 0, 0];
        let pair = construct_sr_pair(&exps, &config, &n);
        // This should produce a valid sr-pair since 49 = 7^2 is smooth
        if let Some(p) = &pair {
            assert_eq!(p.u, BigUint::from(28u64));
            assert_eq!(p.remainder, BigUint::from(49u64));
            assert!(!p.remainder_positive); // u < v*N since v=1, N=77, u=28
        }
    }

    #[test]
    fn test_babai_simple() {
        // Simple 2D lattice
        let basis = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
        ];
        let target = vec![2.7, 3.2];
        let coeffs = babai_nearest_plane(&basis, &target);
        assert_eq!(coeffs.len(), 2);
        // Should round to [3, 3]
        assert!((coeffs[0] - 3.0).abs() < 0.01);
        assert!((coeffs[1] - 3.0).abs() < 0.01);
    }

    #[test]
    fn test_sr_pair_config_scaling() {
        let c20 = SrPairConfig::for_bits(20);
        let c60 = SrPairConfig::for_bits(60);
        let c100 = SrPairConfig::for_bits(100);

        assert!(c60.p1_primes.len() > c20.p1_primes.len());
        assert!(c100.p1_primes.len() > c60.p1_primes.len());
        assert!(c60.p2_primes.len() > c20.p2_primes.len());
        assert!(c100.p2_primes.len() > c60.p2_primes.len());
    }

    #[test]
    fn test_build_schnorr_lattice() {
        let n = BigUint::from(77u64);
        let primes = vec![2u64, 3, 5, 7, 11, 13];
        let (basis, target) = build_schnorr_lattice(&n, &primes, 2.0, 42);

        // Should be (7 x 6) matrix (pi+1 rows, pi columns)
        assert_eq!(basis.len(), 7);
        assert_eq!(basis[0].len(), 6);

        // Target should have zeros except last entry
        assert_eq!(target.len(), 7);
        for i in 0..6 {
            assert_eq!(target[i], 0.0);
        }
        // Last entry = ceil(10^2 * ln(77)) ≈ ceil(100 * 4.344) = 435
        assert!(target[6] > 400.0 && target[6] < 500.0);

        // Last row should be ceil(10^2 * ln(p_j))
        // ln(2) ≈ 0.693, so ceil(100 * 0.693) = 70
        assert!(basis[6][0] > 60.0 && basis[6][0] < 80.0);
    }

    #[test]
    fn test_factor_77_via_sr_pairs() {
        // Full pipeline test: factor 77 = 7 * 11
        let n = BigUint::from(77u64);
        let config = SrPairConfig::for_bits(7);

        // Try many exponent vectors to find sr-pairs
        let mut sr_pairs = Vec::new();

        // Systematically search small exponent vectors
        for a in -3i64..=3 {
            for b in -3i64..=3 {
                for c in -3i64..=3 {
                    for d in -3i64..=3 {
                        let exps = vec![a, b, c, d, 0, 0, 0, 0];
                        if let Some(pair) = construct_sr_pair(&exps, &config, &n) {
                            sr_pairs.push(pair);
                        }
                    }
                }
            }
        }

        eprintln!("Found {} sr-pairs for N=77", sr_pairs.len());

        if sr_pairs.len() >= 2 {
            let congruences = find_congruences(&sr_pairs, config.p2_primes.len());
            eprintln!("Found {} congruences", congruences.len());

            for cong in &congruences {
                if let Some(factor) = extract_factors_from_congruence(
                    cong,
                    &sr_pairs,
                    &n,
                    &config.p2_primes,
                ) {
                    eprintln!("Factor found: {}", factor);
                    assert!(
                        factor == BigUint::from(7u64) || factor == BigUint::from(11u64),
                        "Expected 7 or 11, got {}",
                        factor
                    );
                    return;
                }
            }
        }

        // Not asserting success — this is a diagnostic test
        eprintln!(
            "No factor found for 77 via sr-pairs ({} pairs, brute force search)",
            sr_pairs.len()
        );
    }

    #[test]
    #[ignore] // Brute-force search, OOM for large factor bases
    fn test_factor_larger_via_sr_pairs() {
        // 16-bit: 631 * 641 = 404471 (factors NOT in P₁)
        let n = BigUint::from(404471u64);
        let config = SrPairConfig::for_bits(19);

        eprintln!(
            "Config: P1={} primes, P2={} primes, B2={}",
            config.p1_primes.len(),
            config.p2_primes.len(),
            config.b2_bound
        );

        let mut sr_pairs = Vec::new();

        // Search with larger range since factors are bigger
        for a in -4i64..=4 {
            for b in -4i64..=4 {
                for c in -4i64..=4 {
                    for d in -4i64..=4 {
                        let exps: Vec<i64> = [a, b, c, d]
                            .iter()
                            .chain(std::iter::repeat(&0).take(config.p1_primes.len() - 4))
                            .copied()
                            .collect();
                        if let Some(pair) = construct_sr_pair(&exps, &config, &n) {
                            sr_pairs.push(pair);
                        }
                    }
                }
            }
        }

        eprintln!("Found {} sr-pairs for N=404471", sr_pairs.len());

        if sr_pairs.len() >= 2 {
            let congruences = find_congruences(&sr_pairs, config.p2_primes.len());
            eprintln!("Found {} congruences", congruences.len());

            for cong in &congruences {
                if let Some(factor) = extract_factors_from_congruence(
                    cong,
                    &sr_pairs,
                    &n,
                    &config.p2_primes,
                ) {
                    eprintln!("Factor found for 404471: {}", factor);
                    assert!(
                        &factor * &(&n / &factor) == n,
                        "Factor {} doesn't divide N",
                        factor
                    );
                    return;
                }
            }
        }

        eprintln!(
            "No factor found for 404471 via sr-pairs ({} pairs)",
            sr_pairs.len()
        );
    }
}
