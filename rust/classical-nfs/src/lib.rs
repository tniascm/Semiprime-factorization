//! Number Field Sieve implementation for integer factorization.
//!
//! The GNFS is the most efficient known classical algorithm for factoring
//! large integers. This crate implements the core phases:
//! 1. Polynomial selection (base-m method)
//! 2. Lattice sieving (rational + algebraic sides)
//! 3. Smooth relation collection (with large prime variation)
//! 4. Linear algebra (Gaussian elimination over GF(2))
//! 5. Square root computation and factor extraction
//!
//! # Modules
//!
//! - [`polynomial`] - Polynomial selection (base-m method)
//! - [`sieve`] - Line sieve for rational and algebraic sides
//! - [`relation`] - Smooth relation collection and large prime handling
//! - [`linalg`] - GF(2) linear algebra (Gaussian elimination, Block Lanczos)
//! - [`factor`] - Square root and factor extraction
//! - [`pipeline`] - Full NFS pipeline orchestration

pub mod polynomial;
pub mod sieve;
pub mod relation;
pub mod linalg;
pub mod factor;
pub mod pipeline;

// Re-export the new NFS pipeline entry points
pub use pipeline::{factor_nfs, factor_nfs_with_params, factor_nfs_with_stats, NfsPipelineParams, NfsStats};

use num_bigint::BigUint;
use num_integer::Integer;
use num_traits::{One, Zero, ToPrimitive};
use rayon::prelude::*;

/// NFS configuration parameters.
#[derive(Debug, Clone)]
pub struct NfsParams {
    /// Factor base bound
    pub factor_base_bound: u64,
    /// Sieving range for (a, b) pairs
    pub sieve_range: i64,
    /// Number of threads for sieving
    pub num_threads: usize,
}

impl Default for NfsParams {
    fn default() -> Self {
        Self {
            factor_base_bound: 10_000,
            sieve_range: 100_000,
            num_threads: num_cpus::get().unwrap_or(4),
        }
    }
}

// ---------------------------------------------------------------------------
// NFS Polynomial Selection (base-m method)
// ---------------------------------------------------------------------------

/// A polynomial f(x) = c_d * x^d + c_{d-1} * x^{d-1} + ... + c_0
/// chosen so that f(m) ≡ 0 (mod n) where m = floor(n^(1/(d+1))).
#[derive(Debug, Clone)]
pub struct NfsPolynomial {
    /// Coefficients in order [c_0, c_1, ..., c_d] (low-degree first).
    pub coefficients: Vec<BigUint>,
    /// The base m such that f(m) ≡ 0 (mod n).
    pub m: BigUint,
    /// The degree d of the polynomial.
    pub degree: usize,
}

/// Select a polynomial for NFS using the base-m method.
///
/// Given n and a target degree d, compute m = floor(n^(1/(d+1))) and express
/// n in base m: n = c_d * m^d + c_{d-1} * m^{d-1} + ... + c_0.
/// Then f(x) = c_d * x^d + ... + c_0, with f(m) = n ≡ 0 (mod n).
pub fn select_polynomial(n: &BigUint, degree: usize) -> NfsPolynomial {
    let m = nth_root(n, (degree + 1) as u32);

    // Express n in base m: n = c_d * m^d + ... + c_0
    let mut coefficients = Vec::new();
    let mut remaining = n.clone();

    if m.is_zero() || m == BigUint::one() {
        // Degenerate case: m is 0 or 1, just store n as the constant coeff
        coefficients.push(remaining);
        for _ in 1..=degree {
            coefficients.push(BigUint::zero());
        }
    } else {
        // Standard base-m decomposition
        loop {
            let (quot, rem) = remaining.div_rem(&m);
            coefficients.push(rem);
            remaining = quot;
            if remaining.is_zero() {
                break;
            }
        }
        // Pad to exactly degree+1 coefficients
        while coefficients.len() <= degree {
            coefficients.push(BigUint::zero());
        }
    }

    NfsPolynomial {
        coefficients,
        m,
        degree,
    }
}

/// Evaluate the polynomial f(x) at a given integer x (non-negative).
///
/// f(x) = c_0 + c_1 * x + c_2 * x^2 + ... + c_d * x^d
/// using Horner's method.
pub fn eval_polynomial(poly: &NfsPolynomial, x: i64) -> BigUint {
    let x_abs = BigUint::from(x.unsigned_abs());
    let x_neg = x < 0;

    if poly.coefficients.is_empty() {
        return BigUint::zero();
    }

    // Horner's method, but we must handle negative x carefully.
    // For signed evaluation, we track whether the result is negative.
    // Since this is for norm computation we return the absolute value.
    // We compute using BigUint arithmetic, handling signs manually.

    // For Horner's: result = c_d
    // result = result * x + c_{d-1}
    // ...
    // We'll compute positive and negative parts separately.

    // Actually, since coefficients are BigUint (non-negative), and x can be
    // negative, let's use a signed approach with (positive_part, negative_part).

    // Simpler: evaluate |f(x)| by computing with signed arithmetic.
    // We'll use i128 for small numbers, BigUint path for large.

    // For correctness with BigUint coefficients and signed x, track sign explicitly.
    horner_eval_abs(&poly.coefficients, &x_abs, x_neg)
}

/// Horner's evaluation of polynomial with BigUint coefficients at signed x.
/// Returns |f(x)|.
///
/// coefficients are [c_0, c_1, ..., c_d] (low-degree first).
/// We evaluate c_d * x^d + ... + c_0 = ((c_d * x + c_{d-1}) * x + ...) * x + c_0.
///
/// Since x might be negative, x^k alternates sign. We track a signed accumulator
/// as (magnitude, is_negative).
fn horner_eval_abs(coeffs: &[BigUint], x_abs: &BigUint, x_neg: bool) -> BigUint {
    if coeffs.is_empty() {
        return BigUint::zero();
    }

    let d = coeffs.len() - 1;
    // Start with the highest-degree coefficient (always non-negative)
    let mut mag = coeffs[d].clone();
    let mut neg = false;

    for i in (0..d).rev() {
        // Multiply by x: mag *= |x|, neg ^= x_neg
        mag *= x_abs;
        neg ^= x_neg;

        // Add c_i (always non-negative):
        // If neg: result = c_i - mag
        // If !neg: result = c_i + mag
        if neg {
            if coeffs[i] >= mag {
                mag = &coeffs[i] - &mag;
                neg = false;
            } else {
                mag = &mag - &coeffs[i];
                // neg stays true
            }
        } else {
            mag = &mag + &coeffs[i];
            // neg stays false
        }
    }

    mag
}

/// Compute floor(n^(1/k)) — the integer k-th root of n.
fn nth_root(n: &BigUint, k: u32) -> BigUint {
    if n.is_zero() || k == 0 {
        return BigUint::zero();
    }
    if k == 1 {
        return n.clone();
    }
    if k == 2 {
        return isqrt(n);
    }

    // Newton's method for k-th root: x_{i+1} = ((k-1)*x_i + n / x_i^{k-1}) / k
    let k_big = BigUint::from(k);
    let k_minus_1 = BigUint::from(k - 1);

    // Initial guess: use bit length. n has b bits, so n^(1/k) ~ 2^(b/k)
    let bits = n.bits();
    let init_bits = (bits / k as u64).max(1);
    let mut x = BigUint::one() << init_bits as usize;

    loop {
        // x^(k-1)
        let x_pow = pow_biguint(&x, k - 1);
        if x_pow.is_zero() {
            // Shouldn't happen, but guard against it
            return BigUint::one();
        }
        let x_new = (&k_minus_1 * &x + n / &x_pow) / &k_big;

        if x_new >= x {
            break;
        }
        x = x_new;
    }

    // Verify and adjust (Newton's method for integer roots can overshoot by 1)
    while pow_biguint(&(&x + BigUint::one()), k) <= *n {
        x += BigUint::one();
    }
    while &pow_biguint(&x, k) > n {
        if x.is_zero() {
            break;
        }
        x -= BigUint::one();
    }

    x
}

/// Compute base^exp for BigUint (integer exponent as u32).
fn pow_biguint(base: &BigUint, exp: u32) -> BigUint {
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

// ---------------------------------------------------------------------------
// Homogeneous polynomial evaluation for the algebraic norm
// ---------------------------------------------------------------------------

/// Evaluate the homogeneous polynomial f_hom(a, b) = sum_i c_i * a^i * b^(d-i).
/// Returns the absolute value |f_hom(a, b)| as a BigUint, where a is signed.
fn eval_homogeneous_abs(coeffs: &[BigUint], a: i64, b: i64) -> BigUint {
    if coeffs.is_empty() {
        return BigUint::zero();
    }

    let d = coeffs.len() - 1;
    let a_abs = BigUint::from(a.unsigned_abs());
    let b_abs = BigUint::from(b.unsigned_abs());
    let a_neg = a < 0;
    let b_neg = b < 0;

    // Compute each term: c_i * a^i * b^(d-i)
    // Sign of term: a_neg^i XOR b_neg^(d-i) (since c_i >= 0)
    // Accumulate (positive_sum, negative_sum) then return |positive_sum - negative_sum|.

    let mut positive_sum = BigUint::zero();
    let mut negative_sum = BigUint::zero();

    for i in 0..=d {
        if coeffs[i].is_zero() {
            continue;
        }
        let a_pow = pow_biguint(&a_abs, i as u32);
        let b_pow = pow_biguint(&b_abs, (d - i) as u32);
        let term_mag = &coeffs[i] * &a_pow * &b_pow;

        // Sign: a contributes sign^i, b contributes sign^(d-i)
        let term_neg = (a_neg && (i % 2 == 1)) ^ (b_neg && ((d - i) % 2 == 1));

        if term_neg {
            negative_sum += term_mag;
        } else {
            positive_sum += term_mag;
        }
    }

    if positive_sum >= negative_sum {
        positive_sum - negative_sum
    } else {
        negative_sum - positive_sum
    }
}

// ---------------------------------------------------------------------------
// NFS Rational Sieve
// ---------------------------------------------------------------------------

/// A single NFS relation from the sieving phase.
///
/// For a coprime pair (a, b) with b > 0:
/// - Rational side: a + b*m factors over the rational factor base
/// - Algebraic side: |f_hom(a, b)| (the algebraic norm)
#[derive(Debug, Clone)]
pub struct NfsRelation {
    /// The `a` value of the (a, b) pair.
    pub a: i64,
    /// The `b` value of the (a, b) pair, always > 0.
    pub b: i64,
    /// Exponent vector (mod 2) for the rational side factorization over the factor base.
    pub rational_exponents: Vec<u8>,
    /// The algebraic norm value |f_hom(a, b)|.
    pub algebraic_value: BigUint,
}

/// An extended NFS relation that includes full (unreduced) exponents for reconstruction.
#[derive(Debug, Clone)]
struct NfsFullRelation {
    a: i64,
    b: i64,
    /// Exponent vector mod 2 (for the sign bit + rational side).
    exponents_mod2: Vec<u8>,
    /// Full exponent vector (not reduced, for square-root reconstruction).
    full_exponents: Vec<u32>,
    /// Whether a + b*m was negative (tracked as first element of exponent vector).
    rational_negative: bool,
}

/// Perform rational-side sieving for NFS.
///
/// For each coprime pair (a, b) with 1 <= b <= sieve_range and -sieve_range <= a <= sieve_range,
/// compute the rational norm a + b*m and try to factor it over the factor base.
///
/// Uses rayon to parallelize across b values.
pub fn rational_sieve(
    n: &BigUint,
    poly: &NfsPolynomial,
    factor_base: &[u64],
    sieve_range: i64,
) -> Vec<NfsRelation> {
    let m_i128: i128 = poly
        .m
        .to_u128()
        .expect("m must fit in u128 for rational sieve") as i128;

    let _ = n; // n is implicitly used via poly

    // Parallel sieve across b values
    (1..=sieve_range)
        .into_par_iter()
        .flat_map(|b| {
            let mut local_relations = Vec::new();
            let b_i128 = b as i128;

            for a in -sieve_range..=sieve_range {
                // gcd(|a|, b) must be 1 for a valid coprime pair
                if a == 0 {
                    continue;
                }
                let a_abs = a.unsigned_abs() as u64;
                let b_u64 = b as u64;
                if gcd_u64(a_abs, b_u64) != 1 {
                    continue;
                }

                // Rational side: a + b * m
                let rational_val_signed: i128 = (a as i128) + b_i128 * m_i128;
                if rational_val_signed == 0 {
                    continue;
                }

                let rational_neg = rational_val_signed < 0;
                let rational_abs = rational_val_signed.unsigned_abs() as u64;

                // Try to factor the rational side over the factor base
                if let Some((exp_mod2, _full_exp)) =
                    try_factor_u64_over_base(rational_abs, factor_base)
                {
                    // Compute algebraic norm |f_hom(a, b)|
                    let alg_value =
                        eval_homogeneous_abs(&poly.coefficients, a, b);

                    let _ = rational_neg; // sign tracked in relation

                    local_relations.push(NfsRelation {
                        a,
                        b,
                        rational_exponents: exp_mod2,
                        algebraic_value: alg_value,
                    });
                }
            }

            local_relations
        })
        .collect()
}

/// GCD for u64 values.
fn gcd_u64(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

/// Try to factor a u64 value over the factor base.
/// Returns (exponents_mod2, full_exponents) if the value is smooth.
fn try_factor_u64_over_base(val: u64, factor_base: &[u64]) -> Option<(Vec<u8>, Vec<u32>)> {
    if val == 0 {
        return None;
    }

    let mut remaining = val;
    let mut exponents_mod2 = vec![0u8; factor_base.len()];
    let mut full_exponents = vec![0u32; factor_base.len()];

    for (i, &p) in factor_base.iter().enumerate() {
        while remaining % p == 0 {
            remaining /= p;
            exponents_mod2[i] ^= 1;
            full_exponents[i] += 1;
        }
    }

    if remaining == 1 {
        Some((exponents_mod2, full_exponents))
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// NFS Factor Combining
// ---------------------------------------------------------------------------

/// Perform full rational-side sieving with sign tracking for the linear algebra phase.
/// Returns extended relations with both mod-2 and full exponent vectors.
fn rational_sieve_full(
    poly: &NfsPolynomial,
    factor_base: &[u64],
    sieve_range: i64,
) -> Vec<NfsFullRelation> {
    let m_i128: i128 = poly
        .m
        .to_u128()
        .expect("m must fit in u128 for rational sieve") as i128;

    (1..=sieve_range)
        .into_par_iter()
        .flat_map(|b| {
            let mut local_relations = Vec::new();
            let b_i128 = b as i128;

            for a in -sieve_range..=sieve_range {
                if a == 0 {
                    continue;
                }
                let a_abs = a.unsigned_abs() as u64;
                let b_u64 = b as u64;
                if gcd_u64(a_abs, b_u64) != 1 {
                    continue;
                }

                let rational_val_signed: i128 = (a as i128) + b_i128 * m_i128;
                if rational_val_signed == 0 {
                    continue;
                }

                let rational_neg = rational_val_signed < 0;
                let rational_abs = rational_val_signed.unsigned_abs() as u64;

                if let Some((mut exp_mod2, mut full_exp)) =
                    try_factor_u64_over_base(rational_abs, factor_base)
                {
                    // Prepend a sign bit: 1 if negative, 0 if positive
                    exp_mod2.insert(0, if rational_neg { 1 } else { 0 });
                    full_exp.insert(0, if rational_neg { 1 } else { 0 });

                    local_relations.push(NfsFullRelation {
                        a,
                        b,
                        exponents_mod2: exp_mod2,
                        full_exponents: full_exp,
                        rational_negative: rational_neg,
                    });
                }
            }

            local_relations
        })
        .collect()
}

/// Main NFS factorization entry point.
///
/// Implements a simplified General Number Field Sieve:
/// 1. Select polynomial via base-m method (degree 3)
/// 2. Build rational factor base
/// 3. Sieve for smooth relations on the rational side
/// 4. Find GF(2) dependencies via Gaussian elimination
/// 5. Combine relations to extract factors via gcd
///
/// Falls back to quadratic sieve if NFS fails to find a factor
/// (NFS is asymptotically faster only for large numbers; for small numbers
/// the overhead and probability of success may be lower).
pub fn nfs_factor(n: &BigUint, params: &NfsParams) -> Option<BigUint> {
    let one = BigUint::one();

    // Trivial checks
    if *n <= one {
        return None;
    }
    if n.is_even() {
        return Some(BigUint::from(2u64));
    }

    // Check perfect square
    let s = isqrt(n);
    if &s * &s == *n {
        return Some(s);
    }

    // Step 1: Polynomial selection (degree 3 for small-to-medium numbers)
    let degree = 3;
    let poly = select_polynomial(n, degree);

    // Degenerate polynomial check: if m < 2, NFS won't work well
    if poly.m < BigUint::from(2u64) {
        return quadratic_sieve(n, params.factor_base_bound);
    }

    // Step 2: Build rational factor base — small primes up to the bound
    let factor_base = sieve_primes(params.factor_base_bound);
    if factor_base.is_empty() {
        return quadratic_sieve(n, params.factor_base_bound);
    }

    let fb_size = factor_base.len();
    // We need > fb_size + 1 relations (one extra column for the sign bit)
    let target_relations = fb_size + 12;

    // Step 3: Sieve with increasing range until we have enough relations
    let mut sieve_range = std::cmp::min(params.sieve_range, 500);
    let max_sieve_range = params.sieve_range;
    let mut relations: Vec<NfsFullRelation>;

    loop {
        relations = rational_sieve_full(&poly, &factor_base, sieve_range);

        if relations.len() >= target_relations || sieve_range >= max_sieve_range {
            break;
        }
        // Double the range and retry
        sieve_range = std::cmp::min(sieve_range * 2, max_sieve_range);
    }

    if relations.len() < 2 {
        // Not enough relations; fall back to QS
        return quadratic_sieve(n, params.factor_base_bound);
    }

    // Step 4: Build GF(2) matrix and find dependencies
    // Each row has (1 sign bit) + fb_size exponent bits
    let num_cols = fb_size + 1; // +1 for the sign column
    let matrix: Vec<Vec<u8>> = relations
        .iter()
        .map(|r| r.exponents_mod2.clone())
        .collect();

    let dependencies = find_dependencies_gf2(&matrix, num_cols);

    // Step 5: For each dependency, try to extract a non-trivial factor
    // In the simplified rational-only NFS, we have:
    //   product of (a_i + b_i * m) ≡ y^2 (mod n)  [from the rational side]
    // We also know that a + b*m ≡ a + b*m (mod n), and since m is a root of f mod n,
    // the product on the rational side gives us congruences modulo n.
    //
    // From the dependency: the product of the rational norms is a perfect square.
    // x = product of |a_i + b_i * m| mod n is related to y^2 via the factor base.

    for dep in &dependencies {
        if dep.is_empty() {
            continue;
        }

        // Compute the product of (a + b*m) mod n
        // and the square root from half-exponents
        let mut combined_exponents = vec![0u64; num_cols];
        let mut product_mod_n = BigUint::one();
        let mut sign_count = 0u64;

        for &idx in dep {
            let rel = &relations[idx];

            // Compute |a + b*m| and track sign
            let a_i128 = rel.a as i128;
            let b_i128 = rel.b as i128;
            let m_i128: i128 = poly.m.to_u128().unwrap_or(0) as i128;
            let rational_val = a_i128 + b_i128 * m_i128;

            let abs_val = BigUint::from(rational_val.unsigned_abs());
            product_mod_n = (product_mod_n * &abs_val) % n;

            if rel.rational_negative {
                sign_count += 1;
            }

            for (j, &e) in rel.full_exponents.iter().enumerate() {
                combined_exponents[j] += e as u64;
            }
        }

        // Verify all exponents (including sign) are even
        let all_even = combined_exponents.iter().all(|&e| e % 2 == 0);
        if !all_even {
            continue;
        }

        // The sign must be even (even number of negatives => product is positive)
        if sign_count % 2 != 0 {
            continue;
        }

        // y = product of p_i^(e_i/2) mod n (skip index 0 which is the sign column)
        let mut y_mod_n = BigUint::one();
        for j in 1..num_cols {
            let exp = combined_exponents[j];
            if exp > 0 {
                let half_exp = BigUint::from(exp / 2);
                let p_big = BigUint::from(factor_base[j - 1]); // j-1 because index 0 is sign
                let contribution = p_big.modpow(&half_exp, n);
                y_mod_n = (y_mod_n * contribution) % n;
            }
        }

        // Try gcd(product - y, n) and gcd(product + y, n)
        let x_val = &product_mod_n;

        let diff = if *x_val >= y_mod_n {
            x_val - &y_mod_n
        } else {
            n - ((&y_mod_n - x_val) % n)
        };

        if !diff.is_zero() {
            let factor = diff.gcd(n);
            if factor != one && factor != *n {
                return Some(factor);
            }
        }

        let sum = (x_val + &y_mod_n) % n;
        if !sum.is_zero() {
            let factor = sum.gcd(n);
            if factor != one && factor != *n {
                return Some(factor);
            }
        }
    }

    // NFS didn't find a factor; fall back to quadratic sieve
    quadratic_sieve(n, params.factor_base_bound)
}

/// Generate primes up to `bound` using the Sieve of Eratosthenes.
fn sieve_primes(bound: u64) -> Vec<u64> {
    if bound < 2 {
        return vec![];
    }
    let limit = bound as usize;
    let mut is_prime = vec![true; limit + 1];
    is_prime[0] = false;
    if limit >= 1 {
        is_prime[1] = false;
    }
    let mut p = 2;
    while p * p <= limit {
        if is_prime[p] {
            let mut multiple = p * p;
            while multiple <= limit {
                is_prime[multiple] = false;
                multiple += p;
            }
        }
        p += 1;
    }
    (2..=limit).filter(|&i| is_prime[i]).map(|i| i as u64).collect()
}

/// Compute the integer square root (floor) of a `BigUint`.
fn isqrt(n: &BigUint) -> BigUint {
    if n.is_zero() {
        return BigUint::zero();
    }
    // Newton's method
    let mut x = n.clone();
    let mut y = (&x + BigUint::one()) >> 1u32;
    while y < x {
        x = y.clone();
        y = (&x + n / &x) >> 1u32;
    }
    x
}

/// Ceiling of integer square root: smallest integer r such that r*r >= n.
fn isqrt_ceil(n: &BigUint) -> BigUint {
    let s = isqrt(n);
    if &s * &s == *n {
        s
    } else {
        s + BigUint::one()
    }
}

/// Check if `a` is a quadratic residue mod `p` using Euler's criterion.
/// Returns true if a^((p-1)/2) ≡ 1 (mod p).
fn is_quadratic_residue(a: &BigUint, p: u64) -> bool {
    if p == 2 {
        return true;
    }
    let p_big = BigUint::from(p);
    let a_mod = a % &p_big;
    if a_mod.is_zero() {
        return true; // 0 is trivially a residue
    }
    let exp = BigUint::from((p - 1) / 2);
    let result = a_mod.modpow(&exp, &p_big);
    result == BigUint::one()
}

/// A single smooth relation: Q(x) factors completely over the factor base.
#[derive(Debug, Clone)]
struct SmoothRelation {
    /// The x value such that Q(x) = (x + sqrt_n_ceil)^2 - n
    x_offset: u64,
    /// Exponent vector over the factor base (mod 2 for GF(2) elimination)
    exponents: Vec<u8>,
    /// Full exponent vector (not reduced mod 2, for reconstruction)
    full_exponents: Vec<u32>,
}

/// Try to factor `val` over the factor base. Returns exponent vector if smooth,
/// None otherwise.
fn try_factor_over_base(val: &BigUint, factor_base: &[u64]) -> Option<(Vec<u8>, Vec<u32>)> {
    let mut remaining = val.clone();
    let mut exponents_mod2 = vec![0u8; factor_base.len()];
    let mut full_exponents = vec![0u32; factor_base.len()];

    for (i, &p) in factor_base.iter().enumerate() {
        let p_big = BigUint::from(p);
        while (&remaining % &p_big).is_zero() {
            remaining /= &p_big;
            exponents_mod2[i] ^= 1;
            full_exponents[i] += 1;
        }
    }

    if remaining == BigUint::one() {
        Some((exponents_mod2, full_exponents))
    } else {
        None
    }
}

/// Gaussian elimination over GF(2) to find linear dependencies among rows.
/// Each row is a bitvector of exponents mod 2.
/// Returns sets of row indices whose XOR (sum mod 2) is the zero vector.
fn find_dependencies_gf2(matrix: &[Vec<u8>], num_cols: usize) -> Vec<Vec<usize>> {
    let num_rows = matrix.len();
    if num_rows == 0 || num_cols == 0 {
        return vec![];
    }

    // Augmented matrix: each row tracks which original rows contributed
    // We store (bitvector_of_exponents, set_of_original_row_indices)
    let mut rows: Vec<(Vec<u8>, Vec<usize>)> = matrix
        .iter()
        .enumerate()
        .map(|(i, row)| (row.clone(), vec![i]))
        .collect();

    let mut pivot_row_for_col = vec![None::<usize>; num_cols];

    for col in 0..num_cols {
        // Find a row with a 1 in this column that hasn't been used as pivot
        let pivot = (0..num_rows).find(|&r| {
            rows[r].0[col] == 1 && pivot_row_for_col.iter().all(|p| *p != Some(r))
        });

        if let Some(pivot_idx) = pivot {
            pivot_row_for_col[col] = Some(pivot_idx);

            // Eliminate this column from all other rows
            for r in 0..num_rows {
                if r != pivot_idx && rows[r].0[col] == 1 {
                    let pivot_data = rows[pivot_idx].0.clone();
                    let pivot_history = rows[pivot_idx].1.clone();
                    for c in 0..num_cols {
                        rows[r].0[c] ^= pivot_data[c];
                    }
                    // XOR the history sets (symmetric difference)
                    let mut new_history = rows[r].1.clone();
                    for &idx in &pivot_history {
                        if let Some(pos) = new_history.iter().position(|&x| x == idx) {
                            new_history.remove(pos);
                        } else {
                            new_history.push(idx);
                        }
                    }
                    rows[r].1 = new_history;
                }
            }
        }
    }

    // Rows that are now all-zero represent dependencies
    let mut dependencies = Vec::new();
    for (row_vec, history) in &rows {
        if row_vec.iter().all(|&x| x == 0) && history.len() >= 2 {
            dependencies.push(history.clone());
        }
    }

    dependencies
}

/// Quadratic sieve factorization algorithm.
///
/// Factors `n` by:
/// 1. Building a factor base of primes p <= `factor_base_bound` where `n` is a
///    quadratic residue mod p.
/// 2. Sieving for smooth values of Q(x) = (x + ceil(sqrt(n)))^2 - n.
/// 3. Finding linear dependencies in GF(2) among the exponent vectors.
/// 4. Combining relations to find x^2 ≡ y^2 (mod n) and extracting gcd(x-y, n).
///
/// Practical for numbers up to roughly 60 bits.
pub fn quadratic_sieve(n: &BigUint, factor_base_bound: u64) -> Option<BigUint> {
    let one = BigUint::one();
    let zero = BigUint::zero();

    // Trivial checks
    if *n <= one {
        return None;
    }
    if n.is_even() {
        return Some(BigUint::from(2u64));
    }

    // Check if n is a perfect square
    let sqrt_n = isqrt(n);
    if &sqrt_n * &sqrt_n == *n {
        return Some(sqrt_n);
    }

    // Step 1: ceil(sqrt(n))
    let sqrt_n_ceil = isqrt_ceil(n);

    // Step 2: Build factor base — primes p <= bound where n is a QR mod p
    let all_primes = sieve_primes(factor_base_bound);
    let factor_base: Vec<u64> = all_primes
        .into_iter()
        .filter(|&p| is_quadratic_residue(n, p))
        .collect();

    let fb_size = factor_base.len();
    if fb_size == 0 {
        return None;
    }

    // We need at least fb_size + 1 smooth relations to guarantee a dependency
    let target_relations = fb_size + 10; // a few extra for robustness
    let sieve_range = (target_relations as u64) * 50; // heuristic sieve range

    // Step 3: Sieve for smooth values using parallel chunks
    let chunk_size = 1000u64;
    let num_chunks = (sieve_range + chunk_size - 1) / chunk_size;

    let smooth_relations: Vec<SmoothRelation> = (0..num_chunks)
        .into_par_iter()
        .flat_map(|chunk_idx| {
            let start = chunk_idx * chunk_size;
            let end = std::cmp::min(start + chunk_size, sieve_range);
            let mut local_smooth = Vec::new();

            for x_off in start..end {
                let x = &sqrt_n_ceil + BigUint::from(x_off);
                let x_sq = &x * &x;
                // Q(x) = x^2 - n; since x >= ceil(sqrt(n)), Q(x) >= 0
                if x_sq < *n {
                    continue;
                }
                let q_val = &x_sq - n;
                if q_val.is_zero() {
                    continue;
                }

                if let Some((exp_mod2, full_exp)) = try_factor_over_base(&q_val, &factor_base) {
                    local_smooth.push(SmoothRelation {
                        x_offset: x_off,
                        exponents: exp_mod2,
                        full_exponents: full_exp,
                    });
                }
            }

            local_smooth
        })
        .collect();

    if smooth_relations.len() < 2 {
        return None;
    }

    // Step 4: Build GF(2) matrix and find dependencies
    let matrix: Vec<Vec<u8>> = smooth_relations.iter().map(|r| r.exponents.clone()).collect();
    let dependencies = find_dependencies_gf2(&matrix, fb_size);

    // Step 5: For each dependency, try to extract a non-trivial factor
    for dep in &dependencies {
        // Compute x = product of (x_offset + sqrt_n_ceil) mod n
        // Compute y = sqrt(product of Q values) mod n (using half-exponents)
        let mut combined_exponents = vec![0u64; fb_size];
        let mut x_product = BigUint::one();

        for &idx in dep {
            let rel = &smooth_relations[idx];
            let x_val = &sqrt_n_ceil + BigUint::from(rel.x_offset);
            x_product = (x_product * x_val) % n;

            for (j, &e) in rel.full_exponents.iter().enumerate() {
                combined_exponents[j] += e as u64;
            }
        }

        // The combined exponents should all be even (that's the dependency condition)
        // y = product of p_i^(e_i/2) mod n
        let mut y_product = BigUint::one();
        for (j, &exp) in combined_exponents.iter().enumerate() {
            debug_assert!(exp % 2 == 0, "Exponent must be even in a valid dependency");
            if exp > 0 {
                let half_exp = BigUint::from(exp / 2);
                let p_big = BigUint::from(factor_base[j]);
                let contribution = p_big.modpow(&half_exp, n);
                y_product = (y_product * contribution) % n;
            }
        }

        // gcd(x - y, n) and gcd(x + y, n) might give non-trivial factors
        let diff = if x_product >= y_product {
            &x_product - &y_product
        } else {
            n - ((&y_product - &x_product) % n)
        };

        if diff != zero {
            let factor = diff.gcd(n);
            if factor != one && factor != *n {
                return Some(factor);
            }
        }

        let sum = (&x_product + &y_product) % n;
        if sum != zero {
            let factor = sum.gcd(n);
            if factor != one && factor != *n {
                return Some(factor);
            }
        }
    }

    None
}

mod num_cpus {
    pub fn get() -> Option<usize> {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .ok()
    }
}

// ---------------------------------------------------------------------------
// Tonelli-Shanks algorithm for modular square roots
// ---------------------------------------------------------------------------

/// Compute x such that x^2 ≡ n (mod p), or None if n is not a quadratic residue mod p.
///
/// Uses the Tonelli-Shanks algorithm. Requires p to be an odd prime.
fn tonelli_shanks(n: u64, p: u64) -> Option<u64> {
    if p == 2 {
        return Some(n % 2);
    }
    let n = n % p;
    if n == 0 {
        return Some(0);
    }

    // Check that n is a QR mod p using Euler's criterion: n^((p-1)/2) ≡ 1 (mod p)
    if mod_pow_u64(n, (p - 1) / 2, p) != 1 {
        return None;
    }

    // Factor out powers of 2 from p - 1: p - 1 = q * 2^s with q odd
    let mut q = p - 1;
    let mut s: u32 = 0;
    while q % 2 == 0 {
        q /= 2;
        s += 1;
    }

    if s == 1 {
        // p ≡ 3 (mod 4) — simple case
        let r = mod_pow_u64(n, (p + 1) / 4, p);
        return Some(r);
    }

    // Find a quadratic non-residue z
    let mut z = 2u64;
    while mod_pow_u64(z, (p - 1) / 2, p) != p - 1 {
        z += 1;
        if z >= p {
            return None; // shouldn't happen for a prime p
        }
    }

    let mut m = s;
    let mut c = mod_pow_u64(z, q, p);
    let mut t = mod_pow_u64(n, q, p);
    let mut r = mod_pow_u64(n, (q + 1) / 2, p);

    loop {
        if t == 0 {
            return Some(0);
        }
        if t == 1 {
            return Some(r);
        }

        // Find the least i such that t^(2^i) ≡ 1 (mod p)
        let mut i: u32 = 1;
        let mut temp = mul_mod_u64(t, t, p);
        while temp != 1 {
            temp = mul_mod_u64(temp, temp, p);
            i += 1;
            if i >= m {
                return None; // shouldn't happen if n is a QR
            }
        }

        let b = mod_pow_u64(c, 1u64 << (m - i - 1), p);
        m = i;
        c = mul_mod_u64(b, b, p);
        t = mul_mod_u64(t, c, p);
        r = mul_mod_u64(r, b, p);
    }
}

/// Modular exponentiation: base^exp mod modulus, using u128 intermediates to avoid overflow.
fn mod_pow_u64(mut base: u64, mut exp: u64, modulus: u64) -> u64 {
    if modulus == 1 {
        return 0;
    }
    let mut result: u64 = 1;
    base %= modulus;
    while exp > 0 {
        if exp & 1 == 1 {
            result = mul_mod_u64(result, base, modulus);
        }
        exp >>= 1;
        base = mul_mod_u64(base, base, modulus);
    }
    result
}

/// Multiply two u64 values modulo m, using u128 to avoid overflow.
fn mul_mod_u64(a: u64, b: u64, m: u64) -> u64 {
    ((a as u128 * b as u128) % m as u128) as u64
}

// ---------------------------------------------------------------------------
// Self-Initializing Quadratic Sieve (SIQS)
// ---------------------------------------------------------------------------

/// Parameters for a single SIQS polynomial: Q(x) = (a*x + b)^2 - n.
/// The key property is that a divides Q(x) for all integer x, so we sieve
/// Q(x)/a which produces smaller values and more smooth numbers.
#[derive(Debug, Clone)]
struct SiqsPolynomial {
    /// The `a` coefficient, chosen as a product of factor base primes.
    a: BigUint,
    /// The `b` coefficient, satisfying b^2 ≡ n (mod a).
    b: BigUint,
    /// The value a as u64 (for fast division during sieving).
    a_u64: u64,
}

/// A smooth relation found by SIQS.
#[derive(Debug, Clone)]
struct SiqsSmoothRelation {
    /// The value (a*x + b) whose square minus n gives the smooth value.
    x_val: BigUint,
    /// Exponent vector over the factor base (mod 2 for GF(2) elimination).
    exponents: Vec<u8>,
    /// Full exponent vector (not reduced mod 2, for reconstruction).
    full_exponents: Vec<u32>,
}

/// Generate SIQS polynomials by choosing `a` as products of 2-3 factor base primes.
///
/// For each `a`, find `b` such that b^2 ≡ n (mod a) via CRT on the prime factors of `a`.
/// Returns a list of (a, b) polynomial parameters.
fn generate_siqs_polynomials(
    n: &BigUint,
    factor_base: &[u64],
) -> Vec<SiqsPolynomial> {
    let mut polynomials = Vec::new();

    // Use primes from the factor base that are odd and for which n is a QR.
    // Skip 2 since Tonelli-Shanks is for odd primes.
    let usable_primes: Vec<u64> = factor_base
        .iter()
        .copied()
        .filter(|&p| p > 2 && is_quadratic_residue(n, p))
        .collect();

    if usable_primes.is_empty() {
        return polynomials;
    }

    // Generate polynomials with a = product of 2 primes
    let max_polys_2 = 80usize;
    let mut count = 0;
    'outer2: for i in 0..usable_primes.len() {
        for j in (i + 1)..usable_primes.len() {
            let p1 = usable_primes[i];
            let p2 = usable_primes[j];
            let a_val = p1 as u128 * p2 as u128;
            if a_val > u64::MAX as u128 {
                continue;
            }
            let a_u64 = a_val as u64;
            let a_big = BigUint::from(a_u64);

            if let Some(b_big) = find_b_for_siqs(n, &a_big, &[(p1, 1), (p2, 1)]) {
                polynomials.push(SiqsPolynomial {
                    a: a_big,
                    b: b_big,
                    a_u64,
                });
                count += 1;
                if count >= max_polys_2 {
                    break 'outer2;
                }
            }
        }
    }

    // Generate polynomials with a = product of 3 primes (if we have enough primes)
    if usable_primes.len() >= 3 {
        let max_polys_3 = 40usize;
        let mut count3 = 0;
        'outer3: for i in 0..usable_primes.len() {
            for j in (i + 1)..usable_primes.len() {
                for k in (j + 1)..usable_primes.len() {
                    let p1 = usable_primes[i];
                    let p2 = usable_primes[j];
                    let p3 = usable_primes[k];
                    let a_val = p1 as u128 * p2 as u128 * p3 as u128;
                    if a_val > u64::MAX as u128 {
                        continue;
                    }
                    let a_u64 = a_val as u64;
                    let a_big = BigUint::from(a_u64);

                    if let Some(b_big) =
                        find_b_for_siqs(n, &a_big, &[(p1, 1), (p2, 1), (p3, 1)])
                    {
                        polynomials.push(SiqsPolynomial {
                            a: a_big,
                            b: b_big,
                            a_u64,
                        });
                        count3 += 1;
                        if count3 >= max_polys_3 {
                            break 'outer3;
                        }
                    }
                }
            }
        }
    }

    // Also generate single-prime polynomials for extra coverage
    let max_polys_1 = 30usize;
    let mut count1 = 0;
    for &p in &usable_primes {
        let a_big = BigUint::from(p);
        if let Some(b_big) = find_b_for_siqs(n, &a_big, &[(p, 1)]) {
            polynomials.push(SiqsPolynomial {
                a: a_big,
                b: b_big,
                a_u64: p,
            });
            count1 += 1;
            if count1 >= max_polys_1 {
                break;
            }
        }
    }

    polynomials
}

/// Find b such that b^2 ≡ n (mod a), where a = product of distinct primes.
///
/// Uses Tonelli-Shanks to find square roots mod each prime factor, then CRT
/// to combine them into a solution mod a.
fn find_b_for_siqs(
    n: &BigUint,
    a: &BigUint,
    prime_factors: &[(u64, u32)],
) -> Option<BigUint> {
    if prime_factors.is_empty() {
        return None;
    }

    // For each prime factor p_i, find r_i such that r_i^2 ≡ n (mod p_i)
    let mut residues: Vec<u64> = Vec::new();
    let mut moduli: Vec<u64> = Vec::new();

    for &(p, _exp) in prime_factors {
        let n_mod_p = (n % BigUint::from(p)).to_u64().unwrap_or(0);
        match tonelli_shanks(n_mod_p, p) {
            Some(r) => {
                residues.push(r);
                moduli.push(p);
            }
            None => return None,
        }
    }

    // CRT to combine: find b such that b ≡ r_i (mod p_i) for all i
    let b = crt_combine(&residues, &moduli)?;

    // Verify b^2 ≡ n (mod a) — use BigUint arithmetic for correctness
    let b_big = BigUint::from(b);
    let b_sq_mod_a = (&b_big * &b_big) % a;
    let n_mod_a = n % a;
    if b_sq_mod_a == n_mod_a {
        Some(b_big)
    } else {
        // Try the other root (a - b)
        let a_u64 = a.to_u64().unwrap_or(0);
        if b <= a_u64 {
            let b_alt = a_u64 - b;
            let b_alt_big = BigUint::from(b_alt);
            let b_alt_sq_mod_a = (&b_alt_big * &b_alt_big) % a;
            if b_alt_sq_mod_a == n_mod_a {
                return Some(b_alt_big);
            }
        }
        None
    }
}

/// Chinese Remainder Theorem for small moduli.
/// Given residues r_i and coprime moduli m_i, find x such that x ≡ r_i (mod m_i).
/// Returns x mod (product of m_i).
fn crt_combine(residues: &[u64], moduli: &[u64]) -> Option<u64> {
    if residues.is_empty() || residues.len() != moduli.len() {
        return None;
    }
    if residues.len() == 1 {
        return Some(residues[0] % moduli[0]);
    }

    let mut result: u128 = residues[0] as u128;
    let mut current_mod: u128 = moduli[0] as u128;

    for i in 1..residues.len() {
        let m_i = moduli[i] as u128;
        let r_i = residues[i] as u128;

        // Solve: result + current_mod * t ≡ r_i (mod m_i)
        // current_mod * t ≡ (r_i - result) (mod m_i)
        let diff = if r_i >= result % m_i {
            r_i - result % m_i
        } else {
            m_i - (result % m_i - r_i)
        };

        // t ≡ diff * inverse(current_mod, m_i) (mod m_i)
        let inv = mod_inverse_u128(current_mod % m_i, m_i)?;
        let t = (diff * inv) % m_i;

        result += current_mod * t;
        current_mod *= m_i;
        result %= current_mod;
    }

    // Ensure result fits in u64
    if result <= u64::MAX as u128 {
        Some(result as u64)
    } else {
        None
    }
}

/// Modular multiplicative inverse using extended Euclidean algorithm.
/// Returns a^(-1) mod m, or None if gcd(a, m) != 1.
fn mod_inverse_u128(a: u128, m: u128) -> Option<u128> {
    if m == 0 {
        return None;
    }
    let (g, x) = extended_gcd_i128(a as i128, m as i128);
    if g != 1 {
        return None;
    }
    Some(((x % m as i128 + m as i128) % m as i128) as u128)
}

/// Extended GCD (iterative): returns (gcd, x) such that a*x + b*y = gcd.
fn extended_gcd_i128(a: i128, b: i128) -> (i128, i128) {
    let mut old_r = a;
    let mut r = b;
    let mut old_s: i128 = 1;
    let mut s: i128 = 0;

    while r != 0 {
        let quotient = old_r / r;
        let temp_r = r;
        r = old_r - quotient * r;
        old_r = temp_r;

        let temp_s = s;
        s = old_s - quotient * s;
        old_s = temp_s;
    }

    (old_r, old_s)
}

/// Self-Initializing Quadratic Sieve (SIQS) factorization.
///
/// Improvement over basic QS: uses multiple polynomials Q_i(x) = ((a_i*x + b_i)^2 - n) / a_i
/// where a_i is a product of factor base primes and b_i^2 ≡ n (mod a_i).
/// Since a_i divides Q_i(x) for all x, the sieved values are smaller, yielding more
/// smooth numbers per unit of sieve range.
///
/// The polynomial switching is parallelized across different `a` values using rayon.
pub fn siqs_factor(n: &BigUint, factor_base_bound: u64) -> Option<BigUint> {
    let one = BigUint::one();

    // Trivial checks
    if *n <= one {
        return None;
    }
    if n.is_even() {
        return Some(BigUint::from(2u64));
    }

    // Check perfect square
    let sqrt_n = isqrt(n);
    if &sqrt_n * &sqrt_n == *n {
        return Some(sqrt_n);
    }

    // Build factor base — primes p <= bound where n is a QR mod p
    let all_primes = sieve_primes(factor_base_bound);
    let factor_base: Vec<u64> = all_primes
        .into_iter()
        .filter(|&p| is_quadratic_residue(n, p))
        .collect();

    let fb_size = factor_base.len();
    if fb_size == 0 {
        return None;
    }

    let target_relations = fb_size + 10;

    // Generate SIQS polynomials
    let polynomials = generate_siqs_polynomials(n, &factor_base);

    if polynomials.is_empty() {
        // Fall back to basic QS if we can't generate any SIQS polynomials
        return quadratic_sieve(n, factor_base_bound);
    }

    // Determine sieve range per polynomial — scale with factor base size
    let sieve_per_poly = ((target_relations as u64) * 30).max(2000);

    // Sieve in parallel across polynomials
    let smooth_relations: Vec<SiqsSmoothRelation> = polynomials
        .par_iter()
        .flat_map(|poly| {
            let mut local_smooth = Vec::new();
            let a = &poly.a;
            let b = &poly.b;
            let a_u64 = poly.a_u64;

            for x_signed in -(sieve_per_poly as i64)..=(sieve_per_poly as i64) {
                // Compute v = a*x + b
                // v could be negative if x is very negative, but we need v^2 - n >= 0
                // Actually we need v^2 >= n, so v >= sqrt(n).
                // A simpler approach: compute |v| and check v^2 >= n.

                let ax = if x_signed >= 0 {
                    a * BigUint::from(x_signed as u64)
                } else {
                    let ax_neg = a * BigUint::from((-x_signed) as u64);
                    if ax_neg > *b {
                        // v = b - ax_neg would be negative; v^2 is still valid
                        let v = &ax_neg - b;
                        let v_sq = &v * &v;
                        if v_sq < *n {
                            continue;
                        }
                        let q_full = &v_sq - n;
                        if q_full.is_zero() {
                            continue;
                        }
                        // Q(x) = (v^2 - n) / a
                        let q_div = &q_full / a;
                        // Check that a actually divides q_full
                        if &q_div * a != q_full {
                            // a doesn't divide exactly; try raw q_full
                            if let Some((exp_mod2, full_exp)) =
                                try_factor_over_base(&q_full, &factor_base)
                            {
                                local_smooth.push(SiqsSmoothRelation {
                                    x_val: v,
                                    exponents: exp_mod2,
                                    full_exponents: full_exp,
                                });
                            }
                            continue;
                        }
                        if let Some((exp_mod2, full_exp)) =
                            try_factor_over_base(&q_div, &factor_base)
                        {
                            // We also need to account for the factor of `a` that we
                            // divided out. Since a is a product of factor base primes,
                            // add its exponents.
                            let mut adj_exp_mod2 = exp_mod2;
                            let mut adj_full_exp = full_exp;
                            add_factor_exponents(
                                a_u64,
                                &factor_base,
                                &mut adj_exp_mod2,
                                &mut adj_full_exp,
                            );
                            local_smooth.push(SiqsSmoothRelation {
                                x_val: v,
                                exponents: adj_exp_mod2,
                                full_exponents: adj_full_exp,
                            });
                        }
                        continue;
                    } else {
                        b - &ax_neg
                    }
                };

                let v = if x_signed >= 0 { &ax + b } else { ax.clone() };

                let v_sq = &v * &v;
                if v_sq < *n {
                    continue;
                }
                let q_full = &v_sq - n;
                if q_full.is_zero() {
                    continue;
                }

                // Q(x) = (v^2 - n) / a
                let q_div = &q_full / a;
                if &q_div * a != q_full {
                    // a doesn't divide exactly; try raw q_full instead
                    if let Some((exp_mod2, full_exp)) =
                        try_factor_over_base(&q_full, &factor_base)
                    {
                        local_smooth.push(SiqsSmoothRelation {
                            x_val: v,
                            exponents: exp_mod2,
                            full_exponents: full_exp,
                        });
                    }
                    continue;
                }

                if let Some((exp_mod2, full_exp)) =
                    try_factor_over_base(&q_div, &factor_base)
                {
                    let mut adj_exp_mod2 = exp_mod2;
                    let mut adj_full_exp = full_exp;
                    add_factor_exponents(
                        a_u64,
                        &factor_base,
                        &mut adj_exp_mod2,
                        &mut adj_full_exp,
                    );
                    local_smooth.push(SiqsSmoothRelation {
                        x_val: v,
                        exponents: adj_exp_mod2,
                        full_exponents: adj_full_exp,
                    });
                }
            }

            local_smooth
        })
        .collect();

    // Also collect relations from basic QS sieving for extra coverage
    let sqrt_n_ceil = isqrt_ceil(n);
    let basic_sieve_range = (target_relations as u64) * 50;
    let basic_chunk_size = 1000u64;
    let basic_num_chunks = (basic_sieve_range + basic_chunk_size - 1) / basic_chunk_size;

    let basic_relations: Vec<SiqsSmoothRelation> = (0..basic_num_chunks)
        .into_par_iter()
        .flat_map(|chunk_idx| {
            let start = chunk_idx * basic_chunk_size;
            let end = std::cmp::min(start + basic_chunk_size, basic_sieve_range);
            let mut local_smooth = Vec::new();

            for x_off in start..end {
                let x = &sqrt_n_ceil + BigUint::from(x_off);
                let x_sq = &x * &x;
                if x_sq < *n {
                    continue;
                }
                let q_val = &x_sq - n;
                if q_val.is_zero() {
                    continue;
                }

                if let Some((exp_mod2, full_exp)) = try_factor_over_base(&q_val, &factor_base) {
                    local_smooth.push(SiqsSmoothRelation {
                        x_val: x,
                        exponents: exp_mod2,
                        full_exponents: full_exp,
                    });
                }
            }

            local_smooth
        })
        .collect();

    // Combine all relations
    let mut all_relations: Vec<SiqsSmoothRelation> = smooth_relations;
    all_relations.extend(basic_relations);

    if all_relations.len() < 2 {
        return None;
    }

    // Build GF(2) matrix and find dependencies
    let matrix: Vec<Vec<u8>> = all_relations.iter().map(|r| r.exponents.clone()).collect();
    let dependencies = find_dependencies_gf2(&matrix, fb_size);

    // For each dependency, try to extract a non-trivial factor
    for dep in &dependencies {
        if dep.is_empty() {
            continue;
        }

        let mut combined_exponents = vec![0u64; fb_size];
        let mut x_product = BigUint::one();

        for &idx in dep {
            let rel = &all_relations[idx];
            x_product = (x_product * &rel.x_val) % n;

            for (j, &e) in rel.full_exponents.iter().enumerate() {
                combined_exponents[j] += e as u64;
            }
        }

        // Verify all exponents are even
        if !combined_exponents.iter().all(|&e| e % 2 == 0) {
            continue;
        }

        // y = product of p_i^(e_i/2) mod n
        let mut y_product = BigUint::one();
        for (j, &exp) in combined_exponents.iter().enumerate() {
            if exp > 0 {
                let half_exp = BigUint::from(exp / 2);
                let p_big = BigUint::from(factor_base[j]);
                let contribution = p_big.modpow(&half_exp, n);
                y_product = (y_product * contribution) % n;
            }
        }

        // gcd(x - y, n) and gcd(x + y, n)
        let diff = if x_product >= y_product {
            &x_product - &y_product
        } else {
            n - ((&y_product - &x_product) % n)
        };

        if !diff.is_zero() {
            let factor = diff.gcd(n);
            if factor != one && factor != *n {
                return Some(factor);
            }
        }

        let sum = (&x_product + &y_product) % n;
        if !sum.is_zero() {
            let factor = sum.gcd(n);
            if factor != one && factor != *n {
                return Some(factor);
            }
        }
    }

    // SIQS didn't find a factor; try basic QS as fallback
    quadratic_sieve(n, factor_base_bound)
}

/// Add the exponents from factoring `a_val` over the factor base to the given vectors.
/// Since `a_val` is constructed as a product of factor base primes, this always succeeds.
fn add_factor_exponents(
    a_val: u64,
    factor_base: &[u64],
    exp_mod2: &mut [u8],
    full_exp: &mut [u32],
) {
    let mut remaining = a_val;
    for (i, &p) in factor_base.iter().enumerate() {
        while remaining % p == 0 && remaining > 0 {
            remaining /= p;
            exp_mod2[i] ^= 1;
            full_exp[i] += 1;
        }
        if remaining == 1 {
            break;
        }
    }
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
    fn test_isqrt() {
        assert_eq!(isqrt(&BigUint::from(0u32)), BigUint::zero());
        assert_eq!(isqrt(&BigUint::from(1u32)), BigUint::one());
        assert_eq!(isqrt(&BigUint::from(4u32)), BigUint::from(2u32));
        assert_eq!(isqrt(&BigUint::from(10u32)), BigUint::from(3u32));
        assert_eq!(isqrt(&BigUint::from(100u32)), BigUint::from(10u32));
    }

    #[test]
    fn test_isqrt_ceil() {
        assert_eq!(isqrt_ceil(&BigUint::from(4u32)), BigUint::from(2u32));
        assert_eq!(isqrt_ceil(&BigUint::from(5u32)), BigUint::from(3u32));
        assert_eq!(isqrt_ceil(&BigUint::from(9u32)), BigUint::from(3u32));
    }

    #[test]
    fn test_quadratic_residue() {
        // 2 is a QR mod 7: 3^2 = 9 ≡ 2 (mod 7)
        assert!(is_quadratic_residue(&BigUint::from(2u32), 7));
        // 3 is not a QR mod 7
        assert!(!is_quadratic_residue(&BigUint::from(3u32), 7));
    }

    #[test]
    fn test_quadratic_sieve() {
        let n = BigUint::from(15347u64); // 103 * 149
        let result = quadratic_sieve(&n, 50);
        assert!(result.is_some());
        let f = result.unwrap();
        assert!(n.clone() % &f == BigUint::zero());
        assert!(f != BigUint::one());
        assert!(f != n);
    }

    #[test]
    fn test_quadratic_sieve_small_semiprime() {
        let n = BigUint::from(8051u64); // 83 * 97
        let result = quadratic_sieve(&n, 50);
        assert!(result.is_some());
        let f = result.unwrap();
        assert!(n.clone() % &f == BigUint::zero());
    }

    #[test]
    fn test_quadratic_sieve_even() {
        let n = BigUint::from(100u64);
        let result = quadratic_sieve(&n, 50);
        assert_eq!(result, Some(BigUint::from(2u64)));
    }

    #[test]
    fn test_quadratic_sieve_perfect_square() {
        let n = BigUint::from(10201u64); // 101^2
        let result = quadratic_sieve(&n, 50);
        assert_eq!(result, Some(BigUint::from(101u64)));
    }

    #[test]
    fn test_quadratic_sieve_larger() {
        // Slightly larger: 257 * 263 = 67591
        let n = BigUint::from(67591u64);
        let result = quadratic_sieve(&n, 100);
        assert!(result.is_some());
        let f = result.unwrap();
        assert!(n.clone() % &f == BigUint::zero());
        assert!(f != BigUint::one());
        assert!(f != n);
    }

    // -------------------------------------------------------------------
    // NFS-specific tests
    // -------------------------------------------------------------------

    #[test]
    fn test_nth_root() {
        // 2^10 = 1024; floor(1024^(1/3)) = 10
        assert_eq!(nth_root(&BigUint::from(1024u64), 3), BigUint::from(10u64));
        // floor(1000^(1/3)) = 10
        assert_eq!(nth_root(&BigUint::from(1000u64), 3), BigUint::from(10u64));
        // floor(999^(1/3)) = 9
        assert_eq!(nth_root(&BigUint::from(999u64), 3), BigUint::from(9u64));
        // floor(8^(1/3)) = 2
        assert_eq!(nth_root(&BigUint::from(8u64), 3), BigUint::from(2u64));
        // floor(27^(1/3)) = 3
        assert_eq!(nth_root(&BigUint::from(27u64), 3), BigUint::from(3u64));
        // floor(16^(1/4)) = 2
        assert_eq!(nth_root(&BigUint::from(16u64), 4), BigUint::from(2u64));
        // floor(100^(1/2)) = 10
        assert_eq!(nth_root(&BigUint::from(100u64), 2), BigUint::from(10u64));
    }

    #[test]
    fn test_polynomial_selection() {
        // n = 15347 = 103 * 149
        let n = BigUint::from(15347u64);
        let poly = select_polynomial(&n, 3);

        // Verify degree
        assert_eq!(poly.degree, 3);

        // Verify m = floor(n^(1/4)) for degree 3
        // n^(1/4) = 15347^0.25 ~ 11.12..., so m = 11
        let expected_m = nth_root(&n, 4);
        assert_eq!(poly.m, expected_m);

        // Verify f(m) = n, i.e. the base-m representation reconstructs n
        // f(m) = c_0 + c_1*m + c_2*m^2 + c_3*m^3
        let mut reconstructed = BigUint::zero();
        let mut m_power = BigUint::one();
        for coeff in &poly.coefficients {
            reconstructed += coeff * &m_power;
            m_power *= &poly.m;
        }
        assert_eq!(reconstructed, n, "f(m) must equal n");

        // Verify using eval_polynomial
        let m_i64 = poly.m.to_u64().unwrap() as i64;
        let eval_result = eval_polynomial(&poly, m_i64);
        assert_eq!(eval_result, n, "eval_polynomial(poly, m) must equal n");
    }

    #[test]
    fn test_polynomial_selection_larger() {
        // n = 67591 = 257 * 263
        let n = BigUint::from(67591u64);
        let poly = select_polynomial(&n, 3);

        // Reconstruct n from coefficients
        let mut reconstructed = BigUint::zero();
        let mut m_power = BigUint::one();
        for coeff in &poly.coefficients {
            reconstructed += coeff * &m_power;
            m_power *= &poly.m;
        }
        assert_eq!(reconstructed, n, "f(m) must equal n for 67591");
    }

    #[test]
    fn test_eval_polynomial_simple() {
        // f(x) = 2x^2 + 3x + 5 with m=10 (just for testing eval, not for NFS)
        let poly = NfsPolynomial {
            coefficients: vec![
                BigUint::from(5u64),
                BigUint::from(3u64),
                BigUint::from(2u64),
            ],
            m: BigUint::from(10u64),
            degree: 2,
        };
        // f(1) = 2 + 3 + 5 = 10
        assert_eq!(eval_polynomial(&poly, 1), BigUint::from(10u64));
        // f(2) = 8 + 6 + 5 = 19
        assert_eq!(eval_polynomial(&poly, 2), BigUint::from(19u64));
        // f(0) = 5
        assert_eq!(eval_polynomial(&poly, 0), BigUint::from(5u64));
        // f(-1) = 2 - 3 + 5 = 4
        assert_eq!(eval_polynomial(&poly, -1), BigUint::from(4u64));
    }

    #[test]
    fn test_rational_sieve_produces_relations() {
        let n = BigUint::from(15347u64);
        let poly = select_polynomial(&n, 3);
        let factor_base = sieve_primes(50);

        let relations = rational_sieve(&n, &poly, &factor_base, 20);

        // We should get at least some relations for a small number
        assert!(
            !relations.is_empty(),
            "rational_sieve should produce at least one relation for n=15347"
        );

        // Verify each relation: a + b*m should factor over the factor base
        for rel in &relations {
            assert!(rel.b > 0, "b must be positive");
            assert!(gcd_u64(rel.a.unsigned_abs() as u64, rel.b as u64) == 1, "gcd(|a|,b) must be 1");

            // Verify rational_exponents length matches factor base
            assert_eq!(rel.rational_exponents.len(), factor_base.len());
        }
    }

    #[test]
    fn test_nfs_factor_small() {
        // 15347 = 103 * 149
        let n = BigUint::from(15347u64);
        let params = NfsParams {
            factor_base_bound: 100,
            sieve_range: 500,
            num_threads: 2,
        };
        let result = nfs_factor(&n, &params);
        assert!(result.is_some(), "nfs_factor should find a factor of 15347");
        let f = result.unwrap();
        assert!(
            n.clone() % &f == BigUint::zero(),
            "factor must divide n"
        );
        assert!(f != BigUint::one(), "factor must not be 1");
        assert!(f != n, "factor must not be n itself");
    }

    #[test]
    fn test_nfs_factor_8051() {
        // 8051 = 83 * 97
        let n = BigUint::from(8051u64);
        let params = NfsParams {
            factor_base_bound: 100,
            sieve_range: 500,
            num_threads: 2,
        };
        let result = nfs_factor(&n, &params);
        assert!(result.is_some(), "nfs_factor should find a factor of 8051");
        let f = result.unwrap();
        assert!(n.clone() % &f == BigUint::zero());
        assert!(f != BigUint::one());
        assert!(f != n);
    }

    #[test]
    fn test_nfs_factor_even() {
        let n = BigUint::from(100u64);
        let params = NfsParams::default();
        assert_eq!(nfs_factor(&n, &params), Some(BigUint::from(2u64)));
    }

    #[test]
    fn test_nfs_factor_perfect_square() {
        let n = BigUint::from(10201u64); // 101^2
        let params = NfsParams::default();
        assert_eq!(nfs_factor(&n, &params), Some(BigUint::from(101u64)));
    }

    #[test]
    fn test_homogeneous_eval() {
        // f(x) = x^2 + 1, coefficients = [1, 0, 1]
        // f_hom(a, b) = a^2 + b^2
        let coeffs = vec![
            BigUint::from(1u64),
            BigUint::zero(),
            BigUint::from(1u64),
        ];
        // f_hom(3, 4) = 9 + 16 = 25
        assert_eq!(eval_homogeneous_abs(&coeffs, 3, 4), BigUint::from(25u64));
        // f_hom(-3, 4) = 9 + 16 = 25 (both terms positive since even powers)
        assert_eq!(eval_homogeneous_abs(&coeffs, -3, 4), BigUint::from(25u64));
    }

    // -------------------------------------------------------------------
    // Tonelli-Shanks tests
    // -------------------------------------------------------------------

    #[test]
    fn test_tonelli_shanks() {
        // sqrt(2) mod 7: 3^2 = 9 ≡ 2 (mod 7), 4^2 = 16 ≡ 2 (mod 7)
        let r = tonelli_shanks(2, 7);
        assert!(r.is_some(), "2 is a QR mod 7, should have a sqrt");
        let r = r.unwrap();
        assert!(r == 3 || r == 4, "sqrt(2) mod 7 should be 3 or 4, got {}", r);
        assert_eq!((r * r) % 7, 2, "r^2 mod 7 must equal 2");
    }

    #[test]
    fn test_tonelli_shanks_more() {
        // sqrt(4) mod 7 = 2 or 5
        let r = tonelli_shanks(4, 7).unwrap();
        assert_eq!((r * r) % 7, 4);

        // sqrt(2) mod 17: check that r^2 ≡ 2 (mod 17)
        let r = tonelli_shanks(2, 17).unwrap();
        assert_eq!((r * r) % 17, 2);

        // 3 is not a QR mod 7
        assert!(tonelli_shanks(3, 7).is_none());

        // sqrt(0) mod any prime = 0
        assert_eq!(tonelli_shanks(0, 13), Some(0));
    }

    #[test]
    fn test_tonelli_shanks_p_eq_3_mod_4() {
        // p = 11 ≡ 3 (mod 4) — takes the simple path (s == 1)
        // 3 is a QR mod 11: 5^2 = 25 ≡ 3 (mod 11), 6^2 = 36 ≡ 3 (mod 11)
        let r = tonelli_shanks(3, 11).unwrap();
        assert_eq!((r * r) % 11, 3);
    }

    // -------------------------------------------------------------------
    // SIQS tests
    // -------------------------------------------------------------------

    #[test]
    fn test_siqs_basic() {
        // 15347 = 103 * 149
        let n = BigUint::from(15347u64);
        let result = siqs_factor(&n, 50);
        assert!(result.is_some(), "SIQS should find a factor of 15347");
        let f = result.unwrap();
        assert!(
            n.clone() % &f == BigUint::zero(),
            "factor {} must divide n=15347",
            f
        );
        assert!(f != BigUint::one(), "factor must not be 1");
        assert!(f != n, "factor must not be n itself");
    }

    #[test]
    fn test_siqs_larger() {
        // 67591 = 257 * 263
        let n = BigUint::from(67591u64);
        let result = siqs_factor(&n, 100);
        assert!(result.is_some(), "SIQS should find a factor of 67591");
        let f = result.unwrap();
        assert!(
            n.clone() % &f == BigUint::zero(),
            "factor {} must divide n=67591",
            f
        );
        assert!(f != BigUint::one(), "factor must not be 1");
        assert!(f != n, "factor must not be n itself");
    }

    #[test]
    fn test_siqs_medium() {
        // 48-bit semiprime: 1000003 * 1000033 = 1000036000099
        let n = BigUint::from(1_000_036_000_099u64);
        let result = siqs_factor(&n, 500);
        assert!(
            result.is_some(),
            "SIQS should find a factor of 1000036000099 = 1000003 * 1000033"
        );
        let f = result.unwrap();
        assert!(
            n.clone() % &f == BigUint::zero(),
            "factor {} must divide n=1000036000099",
            f
        );
        assert!(f != BigUint::one(), "factor must not be 1");
        assert!(f != n, "factor must not be n itself");
    }
}
