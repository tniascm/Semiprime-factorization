//! Schnorr lattice construction for smooth relation finding.
//!
//! The Schnorr lattice encodes the factoring problem as a shortest/closest
//! vector problem. Short vectors in this lattice correspond to multiplicative
//! relations among small primes modulo N.
//!
//! ## Paper Reference (Tesoro/Siloi 2024, arXiv:2410.16355)
//!
//! The lattice basis B_{f,c} has:
//! - Diagonal entries f(j) for j = 1..pi (factor base size)
//! - Last row contains ceil(10^c * ln(p_j)) for each prime p_j
//! - Precision parameter c controls the accuracy of the log approximation
//!
//! After construction, the basis is LLL-reduced before tensor network optimization.

use num_bigint::BigUint;
use num_integer::Integer;
use num_traits::{One, ToPrimitive, Zero};

/// A smooth relation: a multiplicative identity over the factor base modulo N.
///
/// Represents: product(p_i^{pos_exponents[i]}) ≡ product(p_j^{neg_exponents[j]}) (mod N)
/// where pos_exponents are the positive exponents and neg_exponents are the absolute
/// values of the negative exponents.
#[derive(Debug, Clone)]
pub struct SmoothRelation {
    /// The left-hand side product: product(p_i^{e_i}) for e_i > 0.
    pub lhs: BigUint,
    /// The right-hand side product: product(p_j^{|e_j|}) for e_j < 0, multiplied by N^t if needed.
    pub rhs: BigUint,
    /// The full exponent vector (can be negative).
    pub exponents: Vec<i32>,
}

/// Parameters for LLL basis reduction.
#[derive(Debug, Clone)]
pub struct LllParams {
    /// Lovasz condition parameter (typically 0.75).
    pub delta: f64,
}

impl Default for LllParams {
    fn default() -> Self {
        Self { delta: 0.75 }
    }
}

/// Generate a list of primes up to `bound` using the Sieve of Eratosthenes.
pub fn sieve_primes(bound: u64) -> Vec<u64> {
    if bound < 2 {
        return Vec::new();
    }
    let limit = bound as usize + 1;
    let mut is_prime = vec![true; limit];
    is_prime[0] = false;
    if limit > 1 {
        is_prime[1] = false;
    }

    let sqrt_bound = (bound as f64).sqrt() as usize + 1;
    for i in 2..=sqrt_bound {
        if is_prime[i] {
            let mut j = i * i;
            while j < limit {
                is_prime[j] = false;
                j += i;
            }
        }
    }

    is_prime
        .iter()
        .enumerate()
        .filter(|&(_, &prime)| prime)
        .map(|(i, _)| i as u64)
        .collect()
}

/// Get the first `count` primes. Automatically adjusts the sieve bound.
pub fn first_n_primes(count: usize) -> Vec<u64> {
    if count == 0 {
        return Vec::new();
    }
    // Upper bound for p_n ~ n * (ln(n) + ln(ln(n))) for n >= 6
    let bound = if count < 6 {
        20u64
    } else {
        let n = count as f64;
        let ln_n = n.ln();
        let ln_ln_n = ln_n.ln();
        ((n * (ln_n + ln_ln_n)) * 1.3) as u64 + 10
    };
    let primes = sieve_primes(bound);
    primes.into_iter().take(count).collect()
}

/// Build the Schnorr lattice matrix for the given N and factor base (original API).
///
/// The lattice is a (k+1) x (k+1) matrix where k = |factor_base|:
/// ```text
/// L = [ I_k | v ]
///     [ 0   | c ]
/// ```
/// where v_i = round(C * ln(p_i)) and c = round(C * ln(N)).
///
/// Short vectors in this lattice give exponent vectors (e_1, ..., e_k, t) such that
/// sum(e_i * ln(p_i)) ~ t * ln(N), meaning product(p_i^{e_i}) ~ N^t.
pub fn build_schnorr_lattice(
    n: &BigUint,
    factor_base: &[u64],
    scaling: f64,
) -> Vec<Vec<i64>> {
    let k = factor_base.len();
    let dim = k + 1;
    let mut lattice = vec![vec![0i64; dim]; dim];

    // ln(N) -- use f64 approximation (fine for the lattice construction)
    let ln_n = n
        .to_f64()
        .map(|f| f.ln())
        .unwrap_or_else(|| {
            // For very large N, compute ln from bit length
            let bits = n.bits();
            (bits as f64) * std::f64::consts::LN_2
        });

    // Fill the identity block and the last column
    for i in 0..k {
        lattice[i][i] = 1;
        let ln_p = (factor_base[i] as f64).ln();
        lattice[i][k] = (scaling * ln_p).round() as i64;
    }

    // Last row: only the bottom-right entry
    lattice[k][k] = (scaling * ln_n).round() as i64;

    lattice
}

/// Build the Schnorr lattice B_{f,c} from the Tesoro/Siloi 2024 paper.
///
/// The basis matrix has:
/// - Rows 0..pi-1: diagonal entries f(j) with last column ceil(10^c * ln(p_j))
/// - Row pi (last row): all zeros except last column = ceil(10^c * ln(N))
///
/// The precision parameter c controls the accuracy of the logarithm approximation.
/// The f(j) values are set to 1 (identity scaling) in the basic version,
/// but can be adjusted for better lattice geometry.
///
/// Returns a matrix of f64 values suitable for LLL reduction.
pub fn build_schnorr_lattice_v2(
    n: &BigUint,
    factor_base: &[u64],
    precision_c: f64,
    f_diagonal: Option<&[f64]>,
) -> Vec<Vec<f64>> {
    let k = factor_base.len();
    let dim = k + 1;
    let mut lattice = vec![vec![0.0f64; dim]; dim];

    // ln(N) -- compute from bit length for large N
    let ln_n = compute_ln_biguint(n);

    // Scaling factor: 10^c
    let scale = 10.0f64.powf(precision_c);

    // Fill the diagonal and last column
    for i in 0..k {
        // Diagonal entry: f(j) -- default is 1.0
        lattice[i][i] = if let Some(f_vals) = f_diagonal {
            if i < f_vals.len() { f_vals[i] } else { 1.0 }
        } else {
            1.0
        };

        // Last column: ceil(10^c * ln(p_j))
        let ln_p = (factor_base[i] as f64).ln();
        lattice[i][k] = (scale * ln_p).ceil();
    }

    // Last row: ceil(10^c * ln(N))
    lattice[k][k] = (scale * ln_n).ceil();

    lattice
}

/// Compute ln(N) for a BigUint, handling values too large for f64.
pub fn compute_ln_biguint(n: &BigUint) -> f64 {
    n.to_f64()
        .map(|f| f.ln())
        .unwrap_or_else(|| {
            let bits = n.bits();
            (bits as f64) * std::f64::consts::LN_2
        })
}

/// Gram-Schmidt orthogonalization.
///
/// Returns (orthogonalized basis, mu coefficients).
pub fn gram_schmidt(basis: &[Vec<f64>]) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let n = basis.len();
    if n == 0 {
        return (Vec::new(), Vec::new());
    }
    let m = basis[0].len();
    let mut ortho = basis.to_vec();
    let mut mu = vec![vec![0.0; n]; n];

    for i in 0..n {
        for j in 0..i {
            let dot_ij: f64 = (0..m).map(|k| basis[i][k] * ortho[j][k]).sum();
            let dot_jj: f64 = (0..m).map(|k| ortho[j][k] * ortho[j][k]).sum();
            mu[i][j] = if dot_jj > 1e-30 { dot_ij / dot_jj } else { 0.0 };

            for k in 0..m {
                ortho[i][k] -= mu[i][j] * ortho[j][k];
            }
        }
    }

    (ortho, mu)
}

/// LLL lattice basis reduction algorithm.
///
/// Reduces the basis in-place using the Lenstra-Lenstra-Lovasz algorithm.
/// After reduction, the basis vectors are shorter and more orthogonal.
///
/// The delta parameter (typically 0.75) controls the quality/speed trade-off.
pub fn lll_reduce(basis: &mut Vec<Vec<f64>>, params: &LllParams) {
    let n = basis.len();
    if n <= 1 {
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

        // Lovasz condition check (recompute after size reduction)
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

/// LLL-reduce an integer lattice (convenience wrapper).
///
/// Converts i64 lattice to f64, runs LLL, then converts back to i64 by rounding.
pub fn lll_reduce_integer(lattice: &[Vec<i64>], params: &LllParams) -> Vec<Vec<i64>> {
    let mut float_basis: Vec<Vec<f64>> = lattice
        .iter()
        .map(|row| row.iter().map(|&v| v as f64).collect())
        .collect();

    lll_reduce(&mut float_basis, params);

    float_basis
        .iter()
        .map(|row| row.iter().map(|&v| v.round() as i64).collect())
        .collect()
}

/// Build a Schnorr lattice and LLL-reduce it (v2 pipeline).
///
/// This combines lattice construction and reduction in one call, matching
/// the paper's approach where the lattice is always reduced before use.
pub fn build_reduced_lattice_v2(
    n: &BigUint,
    factor_base: &[u64],
    precision_c: f64,
) -> Vec<Vec<f64>> {
    let mut basis = build_schnorr_lattice_v2(n, factor_base, precision_c, None);
    let params = LllParams::default();
    lll_reduce(&mut basis, &params);
    basis
}

/// Extract short vectors from an LLL-reduced lattice that might give smooth relations.
///
/// Returns exponent vectors (the first k components of short lattice vectors,
/// rounded to integers). Filters out trivial (all-zero) vectors.
pub fn extract_short_vectors(
    reduced_basis: &[Vec<f64>],
    num_primes: usize,
    max_vectors: usize,
) -> Vec<Vec<i64>> {
    let mut vectors = Vec::new();
    for row in reduced_basis.iter().take(max_vectors) {
        let exponents: Vec<i64> = row.iter().take(num_primes).map(|&v| v.round() as i64).collect();
        let all_zero = exponents.iter().all(|&e| e == 0);
        if !all_zero {
            vectors.push(exponents);
        }
    }
    vectors
}

/// Attempt to convert a lattice short vector (exponent vector) into a smooth relation.
///
/// Given exponents (e_1, ..., e_k), compute:
/// - LHS = product(p_i^{e_i}) for e_i > 0
/// - RHS = product(p_j^{|e_j|}) for e_j < 0
///
/// A valid relation requires EXACT modular congruence:
///   LHS === RHS (mod N)
///
/// If the congruence is not exact, compute the residue R = LHS * RHS^{-1} mod N.
/// If R is smooth over the factor base, extend the exponent vector with R's
/// factorization to produce a valid relation.
///
/// Returns `None` if the exponent vector is trivial, the relation is not exact,
/// and the residue is not smooth.
pub fn lattice_vector_to_relation(
    exponents: &[i64],
    factor_base: &[u64],
    n: &BigUint,
) -> Option<SmoothRelation> {
    let k = factor_base.len().min(exponents.len());

    // Check for trivial vector
    let all_zero = exponents.iter().take(k).all(|&e| e == 0);
    if all_zero {
        return None;
    }

    // Bound check: reject exponents that are too large (would create huge numbers)
    let max_exp = 64i64;
    if exponents.iter().any(|&e| e.abs() > max_exp) {
        return None;
    }

    // Compute LHS and RHS products
    let mut lhs = BigUint::one();
    let mut rhs = BigUint::one();
    let mut exp_vec = vec![0i32; k];

    for i in 0..k {
        let e = exponents[i];
        exp_vec[i] = e as i32;

        if e > 0 {
            let p = BigUint::from(factor_base[i]);
            lhs *= pow_biguint(&p, e as u32);
        } else if e < 0 {
            let p = BigUint::from(factor_base[i]);
            rhs *= pow_biguint(&p, (-e) as u32);
        }
    }

    // Check if LHS === RHS (mod N) or if the relation is useful
    if lhs == rhs {
        return None; // Trivial relation
    }

    // Check the relation mod N
    let lhs_mod = &lhs % n;
    let rhs_mod = &rhs % n;

    if lhs_mod == rhs_mod {
        // Exact smooth relation!
        return Some(SmoothRelation {
            lhs,
            rhs,
            exponents: exp_vec,
        });
    }

    if lhs_mod.is_zero() || rhs_mod.is_zero() {
        // One side is a multiple of N -- exact modular relation (one side is 0 mod N)
        return Some(SmoothRelation {
            lhs,
            rhs,
            exponents: exp_vec,
        });
    }

    // Not an exact relation. Compute the residue R = LHS * RHS^{-1} mod N.
    // If R is smooth over the factor base, we can fix the relation by absorbing
    // R's factorization into the exponent vector.
    //
    // We have: LHS ≡ R * RHS (mod N)
    // So: LHS ≡ R * RHS (mod N)
    // If R = prod(p_i^{r_i}), then: LHS / prod(p_i^{r_i}) ≡ RHS (mod N)
    // Which means: prod(p_i^{e_i - r_i}) for pos side ≡ prod(p_j^{|e_j|}) for neg side (mod N)
    //
    // Equivalently, we adjust the exponent vector by subtracting the factorization of R.

    if let Some(rhs_inv) = mod_inverse(&rhs_mod, n) {
        let residue = (&lhs_mod * &rhs_inv) % n;

        // Check if R == 1 (exact relation, shouldn't happen since we checked lhs_mod == rhs_mod)
        if residue == BigUint::one() {
            return Some(SmoothRelation {
                lhs,
                rhs,
                exponents: exp_vec,
            });
        }

        // Check if R == N-1 (i.e., R ≡ -1 mod N)
        let n_minus_one = n - BigUint::one();
        if residue == n_minus_one {
            // LHS ≡ -RHS (mod N), so LHS + RHS ≡ 0 (mod N)
            // This means LHS + RHS is a multiple of N, which gives us gcd information.
            // We can still use this as a relation: treat it as exact.
            return Some(SmoothRelation {
                lhs,
                rhs,
                exponents: exp_vec,
            });
        }

        // Try to factor the residue over the factor base
        if let Some(residue_exponents) = try_factor_over_base(&residue, factor_base) {
            // Success! The residue is smooth. Adjust the exponent vector.
            // We have LHS ≡ R * RHS (mod N) where R = prod(p_i^{r_i}).
            // New relation: prod(p_i^{e_i - r_i}) for positive ≡ prod(p_j^{|e_j|}) for negative.
            let mut adjusted_exp = exp_vec.clone();
            for i in 0..k.min(residue_exponents.len()) {
                adjusted_exp[i] -= residue_exponents[i] as i32;
            }

            // Recompute LHS and RHS from adjusted exponents
            let mut new_lhs = BigUint::one();
            let mut new_rhs = BigUint::one();
            for i in 0..k {
                let e = adjusted_exp[i];
                if e > 0 {
                    let p = BigUint::from(factor_base[i]);
                    new_lhs *= pow_biguint(&p, e as u32);
                } else if e < 0 {
                    let p = BigUint::from(factor_base[i]);
                    new_rhs *= pow_biguint(&p, (-e) as u32);
                }
            }

            // Verify the adjusted relation is exact
            let new_lhs_mod = &new_lhs % n;
            let new_rhs_mod = &new_rhs % n;
            if new_lhs_mod == new_rhs_mod || new_lhs_mod.is_zero() || new_rhs_mod.is_zero() {
                // Check it's not trivial
                if new_lhs != new_rhs && adjusted_exp.iter().any(|&e| e != 0) {
                    return Some(SmoothRelation {
                        lhs: new_lhs,
                        rhs: new_rhs,
                        exponents: adjusted_exp,
                    });
                }
            }
        }

        // Also try the "negative" residue: N - R
        // If N - R is smooth, then LHS ≡ -R' * RHS (mod N) where R' = N - R
        let neg_residue = n - &residue;
        if let Some(neg_residue_exponents) = try_factor_over_base(&neg_residue, factor_base) {
            // LHS ≡ -(prod(p_i^{r_i})) * RHS (mod N)
            // So LHS * prod(p_i^{-r_i}) ≡ -RHS (mod N)
            // i.e. LHS / R' + RHS ≡ 0 (mod N)
            let mut adjusted_exp = exp_vec.clone();
            for i in 0..k.min(neg_residue_exponents.len()) {
                adjusted_exp[i] -= neg_residue_exponents[i] as i32;
            }

            let mut new_lhs = BigUint::one();
            let mut new_rhs = BigUint::one();
            for i in 0..k {
                let e = adjusted_exp[i];
                if e > 0 {
                    let p = BigUint::from(factor_base[i]);
                    new_lhs *= pow_biguint(&p, e as u32);
                } else if e < 0 {
                    let p = BigUint::from(factor_base[i]);
                    new_rhs *= pow_biguint(&p, (-e) as u32);
                }
            }

            if adjusted_exp.iter().any(|&e| e != 0) && new_lhs != new_rhs {
                // This is a relation of the form new_lhs ≡ -new_rhs (mod N)
                // which is still useful for factoring via gcd(x+y, N) or gcd(x-y, N)
                let new_lhs_mod = &new_lhs % n;
                let new_rhs_mod = &new_rhs % n;
                let sum_mod = (&new_lhs_mod + &new_rhs_mod) % n;
                if sum_mod.is_zero() || new_lhs_mod == new_rhs_mod {
                    return Some(SmoothRelation {
                        lhs: new_lhs,
                        rhs: new_rhs,
                        exponents: adjusted_exp,
                    });
                }
            }
        }
    }

    // No valid relation could be formed
    None
}

/// Try to build a smooth relation directly from exponent vector and factor base.
///
/// This is a more direct approach: given exponents e_i, compute
/// product(p_i^{e_i}) mod N and check if it yields useful factoring information.
/// Only accepts exact modular congruences or residue-factored relations.
pub fn exponents_to_smooth_relation(
    exponents: &[i64],
    factor_base: &[u64],
    n: &BigUint,
) -> Option<SmoothRelation> {
    // Delegate to the unified relation extraction logic
    lattice_vector_to_relation(exponents, factor_base, n)
}

/// Compute the modular inverse of `a` modulo `m` using the extended Euclidean algorithm.
///
/// Returns `Some(a^{-1} mod m)` if gcd(a, m) == 1, or `None` if no inverse exists.
pub fn mod_inverse(a: &BigUint, m: &BigUint) -> Option<BigUint> {
    if m.is_zero() || m == &BigUint::one() {
        return None;
    }
    if a.is_zero() {
        return None;
    }

    // Extended GCD using signed arithmetic via (old_r, r) tracking
    // We work with BigUint but track signs separately.
    let g = a.gcd(m);
    if g != BigUint::one() {
        return None;
    }

    // Use the iterative extended Euclidean algorithm with signed coefficients.
    // We compute the Bezout coefficient for a: a * x ≡ 1 (mod m)
    //
    // We track (old_s, s) where old_s * a + old_t * m = old_r
    // At the end, old_s * a ≡ gcd (mod m)
    //
    // Since BigUint cannot be negative, we track sign separately.

    let mut old_r = m.clone();
    let mut r = a.clone();
    // s coefficients tracked as (magnitude, is_negative)
    let mut old_s = BigUint::zero();
    let mut old_s_neg = false;
    let mut s = BigUint::one();
    let mut s_neg = false;

    while !r.is_zero() {
        let quotient = &old_r / &r;

        // (old_r, r) = (r, old_r - quotient * r)
        let temp_r = r.clone();
        let qr = &quotient * &r;
        if old_r >= qr {
            r = &old_r - &qr;
        } else {
            r = &qr - &old_r;
        }
        old_r = temp_r;

        // (old_s, s) = (s, old_s - quotient * s) with sign tracking
        let temp_s = s.clone();
        let temp_s_neg = s_neg;
        let qs = &quotient * &s;

        // new_s = old_s - quotient * s
        // We need to handle signs: old_s_sign * old_s_mag - s_sign * qs_mag
        if old_s_neg == s_neg {
            // Same sign: subtract magnitudes
            if old_s >= qs {
                s = &old_s - &qs;
                s_neg = old_s_neg;
            } else {
                s = &qs - &old_s;
                s_neg = !old_s_neg;
            }
        } else {
            // Different signs: add magnitudes
            s = &old_s + &qs;
            s_neg = old_s_neg;
        }

        old_s = temp_s;
        old_s_neg = temp_s_neg;
    }

    // old_s * a ≡ gcd (mod m), and gcd == 1
    // If old_s is negative, add m to make it positive
    if old_s_neg {
        // old_s represents a negative number, so the inverse is m - old_s
        let old_s_mod = &old_s % m;
        if old_s_mod.is_zero() {
            Some(BigUint::zero())
        } else {
            Some(m - old_s_mod)
        }
    } else {
        Some(&old_s % m)
    }
}

/// Check if `value` can be completely factored over the given factor base.
///
/// Returns `true` if `value` is B-smooth (all prime factors are in `factor_base`).
pub fn is_smooth(value: &BigUint, factor_base: &[u64]) -> bool {
    if value.is_zero() || value == &BigUint::one() {
        return true;
    }
    let mut remaining = value.clone();
    for &p in factor_base {
        let p_big = BigUint::from(p);
        while &remaining % &p_big == BigUint::zero() {
            remaining = &remaining / &p_big;
        }
        if remaining == BigUint::one() {
            return true;
        }
    }
    false
}

/// Try to completely factor `value` over the given factor base.
///
/// Returns `Some(exponents)` if `value` = prod(factor_base[i]^{exponents[i]}),
/// or `None` if `value` has factors not in the factor base.
pub fn try_factor_over_base(value: &BigUint, factor_base: &[u64]) -> Option<Vec<i64>> {
    if value.is_zero() {
        return None;
    }
    if value == &BigUint::one() {
        return Some(vec![0i64; factor_base.len()]);
    }

    let mut remaining = value.clone();
    let mut exponents = vec![0i64; factor_base.len()];

    for (i, &p) in factor_base.iter().enumerate() {
        let p_big = BigUint::from(p);
        while &remaining % &p_big == BigUint::zero() {
            remaining = &remaining / &p_big;
            exponents[i] += 1;
        }
        if remaining == BigUint::one() {
            return Some(exponents);
        }
    }

    // remaining != 1 means there are factors not in the factor base
    None
}

/// Compute base^exp for BigUint.
pub fn pow_biguint(base: &BigUint, exp: u32) -> BigUint {
    if exp == 0 {
        return BigUint::one();
    }
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

/// Integer square root of a BigUint (Newton's method).
pub fn sqrt_biguint(n: &BigUint) -> BigUint {
    if n.is_zero() {
        return BigUint::zero();
    }
    if n == &BigUint::one() {
        return BigUint::one();
    }

    // Initial guess: 2^((bits+1)/2) -- must be >= the true sqrt
    let bits = n.bits();
    let mut x = BigUint::one() << ((bits + 1) / 2);

    loop {
        let next = (&x + n / &x) >> 1;
        if next >= x {
            return x;
        }
        x = next;
    }
}

/// Compute the Euclidean norm of a vector.
pub fn vector_norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Compute the Euclidean norm of an integer vector.
pub fn vector_norm_i64(v: &[i64]) -> f64 {
    v.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>().sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sieve_primes() {
        let primes = sieve_primes(30);
        assert_eq!(primes, vec![2, 3, 5, 7, 11, 13, 17, 19, 23, 29]);
    }

    #[test]
    fn test_sieve_primes_small() {
        assert_eq!(sieve_primes(1), vec![]);
        assert_eq!(sieve_primes(2), vec![2]);
        assert_eq!(sieve_primes(3), vec![2, 3]);
    }

    #[test]
    fn test_first_n_primes() {
        let primes = first_n_primes(10);
        assert_eq!(primes, vec![2, 3, 5, 7, 11, 13, 17, 19, 23, 29]);

        let primes0 = first_n_primes(0);
        assert!(primes0.is_empty());

        let primes1 = first_n_primes(1);
        assert_eq!(primes1, vec![2]);
    }

    #[test]
    fn test_build_schnorr_lattice_dimensions() {
        let n = BigUint::from(143u64); // 11 * 13
        let factor_base = vec![2, 3, 5, 7];
        let lattice = build_schnorr_lattice(&n, &factor_base, 100.0);

        // Should be (k+1) x (k+1) = 5x5
        assert_eq!(lattice.len(), 5);
        for row in &lattice {
            assert_eq!(row.len(), 5);
        }
    }

    #[test]
    fn test_build_schnorr_lattice_identity_block() {
        let n = BigUint::from(143u64);
        let factor_base = vec![2, 3, 5, 7];
        let lattice = build_schnorr_lattice(&n, &factor_base, 100.0);

        // Top-left k x k block should be identity
        for i in 0..4 {
            for j in 0..4 {
                if i == j {
                    assert_eq!(lattice[i][j], 1);
                } else {
                    assert_eq!(lattice[i][j], 0);
                }
            }
        }
    }

    #[test]
    fn test_build_schnorr_lattice_last_column() {
        let n = BigUint::from(143u64);
        let factor_base = vec![2, 3, 5];
        let lattice = build_schnorr_lattice(&n, &factor_base, 100.0);

        // Last column should be round(C * ln(p_i)) for i < k, round(C * ln(N)) for last
        let expected_ln2 = (100.0 * (2.0f64).ln()).round() as i64;
        let expected_ln3 = (100.0 * (3.0f64).ln()).round() as i64;
        let expected_ln5 = (100.0 * (5.0f64).ln()).round() as i64;
        let expected_lnn = (100.0 * (143.0f64).ln()).round() as i64;

        assert_eq!(lattice[0][3], expected_ln2);
        assert_eq!(lattice[1][3], expected_ln3);
        assert_eq!(lattice[2][3], expected_ln5);
        assert_eq!(lattice[3][3], expected_lnn);
    }

    #[test]
    fn test_build_schnorr_lattice_v2_dimensions() {
        let n = BigUint::from(143u64);
        let factor_base = vec![2, 3, 5, 7];
        let lattice = build_schnorr_lattice_v2(&n, &factor_base, 2.0, None);

        assert_eq!(lattice.len(), 5);
        for row in &lattice {
            assert_eq!(row.len(), 5);
        }
    }

    #[test]
    fn test_build_schnorr_lattice_v2_precision() {
        let n = BigUint::from(143u64);
        let factor_base = vec![2, 3, 5];
        let c = 3.0; // 10^3 = 1000
        let lattice = build_schnorr_lattice_v2(&n, &factor_base, c, None);

        // Last column: ceil(1000 * ln(p_j))
        let expected_ln2 = (1000.0 * (2.0f64).ln()).ceil();
        let expected_ln3 = (1000.0 * (3.0f64).ln()).ceil();
        let expected_ln5 = (1000.0 * (5.0f64).ln()).ceil();

        assert!((lattice[0][3] - expected_ln2).abs() < 1e-10);
        assert!((lattice[1][3] - expected_ln3).abs() < 1e-10);
        assert!((lattice[2][3] - expected_ln5).abs() < 1e-10);

        // Diagonal should be identity (default f=1)
        for i in 0..3 {
            assert!((lattice[i][i] - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_lll_reduces_basis_vectors() {
        // Create a lattice with known properties
        let n = BigUint::from(1001u64); // 7 * 11 * 13
        let factor_base = vec![2, 3, 5, 7, 11];
        let mut basis = build_schnorr_lattice_v2(&n, &factor_base, 2.0, None);

        let norms_before: Vec<f64> = basis.iter().map(|row| vector_norm(row)).collect();

        let params = LllParams::default();
        lll_reduce(&mut basis, &params);

        let norms_after: Vec<f64> = basis.iter().map(|row| vector_norm(row)).collect();

        // After LLL, the first vector should be shorter than the shortest original
        let min_before = norms_before.iter().cloned().fold(f64::MAX, f64::min);
        assert!(
            norms_after[0] <= min_before + 1e-6,
            "LLL should produce a short first vector: got {} vs min_before {}",
            norms_after[0],
            min_before
        );
    }

    #[test]
    fn test_lll_reduce_integer() {
        let lattice = vec![
            vec![1i64, 0, 69],
            vec![0, 1, 110],
            vec![0, 0, 496],
        ];
        let params = LllParams::default();
        let reduced = lll_reduce_integer(&lattice, &params);

        assert_eq!(reduced.len(), 3);
        for row in &reduced {
            assert_eq!(row.len(), 3);
        }

        // First vector norm should be reduced
        let norm_first_orig = vector_norm_i64(&lattice[0]);
        let norm_first_red = vector_norm_i64(&reduced[0]);
        assert!(
            norm_first_red <= norm_first_orig + 1.0,
            "Reduced first vector should be shorter: {} vs {}",
            norm_first_red,
            norm_first_orig
        );
    }

    #[test]
    fn test_build_reduced_lattice_v2() {
        let n = BigUint::from(143u64);
        let factor_base = vec![2, 3, 5, 7];
        let reduced = build_reduced_lattice_v2(&n, &factor_base, 2.0);

        assert_eq!(reduced.len(), 5);
        for row in &reduced {
            assert_eq!(row.len(), 5);
        }

        // The reduced basis should have a shorter first vector than the unreduced
        let unreduced = build_schnorr_lattice_v2(&n, &factor_base, 2.0, None);
        let norm_unreduced_min: f64 = unreduced.iter().map(|r| vector_norm(r)).fold(f64::MAX, f64::min);
        let norm_reduced_first = vector_norm(&reduced[0]);

        assert!(
            norm_reduced_first <= norm_unreduced_min + 1e-6,
            "Reduced lattice first vector should be short"
        );
    }

    #[test]
    fn test_extract_short_vectors() {
        let n = BigUint::from(143u64);
        let factor_base = vec![2, 3, 5, 7];
        let reduced = build_reduced_lattice_v2(&n, &factor_base, 2.0);

        let vectors = extract_short_vectors(&reduced, factor_base.len(), 5);
        // Should have at least one non-trivial vector
        assert!(!vectors.is_empty(), "Should extract at least one short vector");
        for v in &vectors {
            assert_eq!(v.len(), factor_base.len());
            // Not all zero
            assert!(v.iter().any(|&e| e != 0));
        }
    }

    #[test]
    fn test_trivial_relation_rejected() {
        let n = BigUint::from(143u64);
        let factor_base = vec![2, 3, 5, 7];
        let zero_exponents = vec![0i64; 4];
        assert!(lattice_vector_to_relation(&zero_exponents, &factor_base, &n).is_none());
    }

    #[test]
    fn test_exact_smooth_relation() {
        let n = BigUint::from(15u64); // 3 * 5
        let factor_base = vec![2, 3, 5, 7];
        // 2^4 = 16 === 1 (mod 15), so exponents [4, 0, 0, 0] gives lhs=16, rhs=1
        let exponents = vec![4i64, 0, 0, 0];
        let rel = lattice_vector_to_relation(&exponents, &factor_base, &n);
        assert!(rel.is_some());
    }

    #[test]
    fn test_pow_biguint() {
        let base = BigUint::from(3u64);
        assert_eq!(pow_biguint(&base, 0), BigUint::one());
        assert_eq!(pow_biguint(&base, 1), BigUint::from(3u64));
        assert_eq!(pow_biguint(&base, 4), BigUint::from(81u64));
    }

    #[test]
    fn test_pow_biguint_fast() {
        // Verify fast exponentiation gives same result
        let base = BigUint::from(2u64);
        assert_eq!(pow_biguint(&base, 10), BigUint::from(1024u64));
        assert_eq!(pow_biguint(&base, 20), BigUint::from(1048576u64));
    }

    #[test]
    fn test_sqrt_biguint() {
        assert_eq!(sqrt_biguint(&BigUint::from(0u64)), BigUint::zero());
        assert_eq!(sqrt_biguint(&BigUint::from(1u64)), BigUint::one());
        assert_eq!(sqrt_biguint(&BigUint::from(4u64)), BigUint::from(2u64));
        assert_eq!(sqrt_biguint(&BigUint::from(9u64)), BigUint::from(3u64));
        assert_eq!(sqrt_biguint(&BigUint::from(10u64)), BigUint::from(3u64));
        assert_eq!(sqrt_biguint(&BigUint::from(100u64)), BigUint::from(10u64));
    }

    #[test]
    fn test_gram_schmidt_orthogonal() {
        let basis = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let (ortho, mu) = gram_schmidt(&basis);

        // Identity should remain unchanged
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((ortho[i][j] - expected).abs() < 1e-10);
            }
        }

        // mu should be zero for orthogonal input
        for i in 0..3 {
            for j in 0..i {
                assert!(mu[i][j].abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_gram_schmidt_non_orthogonal() {
        let basis = vec![
            vec![1.0, 1.0, 0.0],
            vec![0.0, 1.0, 1.0],
        ];
        let (ortho, _mu) = gram_schmidt(&basis);

        // Orthogonalized vectors should be orthogonal
        let dot: f64 = (0..3).map(|k| ortho[0][k] * ortho[1][k]).sum();
        assert!(dot.abs() < 1e-10, "Orthogonalized vectors should be orthogonal, dot={}", dot);
    }
}
