//! Hecke trace computation and bounds.
//!
//! For a prime l not dividing N, the trace of the Hecke operator T_l
//! on S_2(Gamma_0(N)) can be computed via the Eichler-Selberg trace formula.
//!
//! For our attack, we use:
//! 1. Dimension formulas to compute dim S_2(Gamma_0(d)) for d | N
//! 2. Ramanujan-Petersson bounds on individual eigenvalues
//! 3. The old-new decomposition structure
//!
//! This module provides both u64 and BigUint versions of all formulas.
//! The BigUint versions handle arbitrary-precision semiprimes (64+ bits).

use num_bigint::{BigInt, BigUint};
use num_integer::Integer;
use num_traits::{One, Zero, ToPrimitive};

// ============================================================
// BigUint versions of all dimension formulas
// ============================================================

/// Dimension of S_2(Gamma_0(N)) for any N, using BigUint arithmetic.
/// Uses exact integer arithmetic: genus = (12 + mu - 3*nu2 - 4*nu3 - 6*cusps) / 12
pub fn dim_s2_big(n: &BigUint) -> BigUint {
    let one = BigUint::one();
    if *n <= one {
        return BigUint::zero();
    }

    let mu = psi_big(n);
    let nu2 = count_elliptic_2_big(n);
    let nu3 = count_elliptic_3_big(n);
    let cusps = count_cusps_big(n);

    // genus = 1 + mu/12 - nu2/4 - nu3/3 - cusps/2
    // Exact: numerator = 12 + mu - 3*nu2 - 4*nu3 - 6*cusps
    // genus = numerator / 12
    // We work in BigInt (signed) to handle potential negative intermediates.
    let twelve = BigInt::from(12u64);
    let mu_i = BigInt::from(mu);
    let nu2_i = BigInt::from(nu2);
    let nu3_i = BigInt::from(nu3);
    let cusps_i = BigInt::from(cusps);

    let numerator = &twelve + &mu_i - BigInt::from(3u64) * &nu2_i
        - BigInt::from(4u64) * &nu3_i - BigInt::from(6u64) * &cusps_i;

    if numerator <= BigInt::zero() {
        BigUint::zero()
    } else {
        let genus = numerator / twelve;
        genus.to_biguint().unwrap_or_else(|| BigUint::zero())
    }
}

/// Psi(N) = N * prod_{p|N} (1 + 1/p) -- the index of Gamma_0(N) in SL_2(Z).
/// BigUint version. Uses wheel factorization (mod 6) for speed.
pub fn psi_big(n: &BigUint) -> BigUint {
    let mut result = n.clone();
    let mut temp = n.clone();
    let mut factors: Vec<BigUint> = Vec::new();

    // Check 2 and 3 first
    let two = BigUint::from(2u64);
    let three = BigUint::from(3u64);
    if (&temp % &two).is_zero() {
        factors.push(two.clone());
        while (&temp % &two).is_zero() {
            temp /= &two;
        }
    }
    if (&temp % &three).is_zero() {
        factors.push(three.clone());
        while (&temp % &three).is_zero() {
            temp /= &three;
        }
    }

    // Wheel mod 6: check p = 6k-1 and p = 6k+1
    let mut p = 5u64;
    while BigUint::from(p) * BigUint::from(p) <= temp {
        let p_big = BigUint::from(p);
        if (&temp % &p_big).is_zero() {
            factors.push(p_big.clone());
            while (&temp % &p_big).is_zero() {
                temp /= &p_big;
            }
        }
        let p2 = p + 2;
        let p2_big = BigUint::from(p2);
        if (&temp % &p2_big).is_zero() {
            factors.push(p2_big.clone());
            while (&temp % &p2_big).is_zero() {
                temp /= &p2_big;
            }
        }
        p += 6;
    }
    if temp > BigUint::one() {
        factors.push(temp);
    }

    // result = N * prod_{p|N} (p+1)/p
    for p in &factors {
        result = &result / p * (p + BigUint::one());
    }
    result
}

/// Compute psi(N) given the prime factorization of N.
/// Much faster than psi_big when factors are already known.
pub fn psi_from_factors(n: &BigUint, prime_factors: &[BigUint]) -> BigUint {
    let mut result = n.clone();
    for p in prime_factors {
        result = &result / p * (p + BigUint::one());
    }
    result
}

/// Compute dim S_2(Gamma_0(N)) given the prime factors of N.
/// Avoids re-factoring N, which is O(sqrt(N)).
pub fn dim_s2_from_factors(n: &BigUint, prime_factors: &[BigUint]) -> BigUint {
    let one = BigUint::one();
    if *n <= one {
        return BigUint::zero();
    }

    let mu = psi_from_factors(n, prime_factors);
    let nu2 = count_elliptic_from_factors(n, prime_factors, -1);
    let nu3 = count_elliptic_from_factors(n, prime_factors, -3);
    let cusps = count_cusps_from_factors(n, prime_factors);

    let twelve = BigInt::from(12u64);
    let mu_i = BigInt::from(mu);
    let nu2_i = BigInt::from(nu2);
    let nu3_i = BigInt::from(nu3);
    let cusps_i = BigInt::from(cusps);

    let numerator = &twelve + &mu_i - BigInt::from(3u64) * &nu2_i
        - BigInt::from(4u64) * &nu3_i - BigInt::from(6u64) * &cusps_i;

    if numerator <= BigInt::zero() {
        BigUint::zero()
    } else {
        let genus = numerator / twelve;
        genus.to_biguint().unwrap_or_else(|| BigUint::zero())
    }
}

/// Count elliptic points given known prime factors.
fn count_elliptic_from_factors(n: &BigUint, prime_factors: &[BigUint], disc: i64) -> BigUint {
    let check_mod = if disc == -1 {
        BigUint::from(4u64)
    } else {
        BigUint::from(9u64)
    };
    if (n % &check_mod).is_zero() {
        return BigUint::zero();
    }

    let skip_p = if disc == -1 {
        BigUint::from(2u64)
    } else {
        BigUint::from(3u64)
    };
    let mut result: i64 = 1;
    for p in prime_factors {
        let legendre = if *p == skip_p {
            0
        } else {
            kronecker_symbol_big(&BigInt::from(disc), p)
        };
        result *= 1 + legendre;
    }
    if result < 0 {
        BigUint::zero()
    } else {
        BigUint::from(result as u64)
    }
}

/// Count cusps given known prime factors.
fn count_cusps_from_factors(n: &BigUint, prime_factors: &[BigUint]) -> BigUint {
    let divisors = divisors_from_prime_factors(n, prime_factors);
    let mut count = BigUint::zero();
    for d in &divisors {
        let nd = n / d;
        count += euler_phi_big(&gcd_big(d, &nd));
    }
    count
}

/// Enumerate all divisors of N from its prime factorization.
fn divisors_from_prime_factors(n: &BigUint, prime_factors: &[BigUint]) -> Vec<BigUint> {
    let mut divs = vec![BigUint::one()];
    let mut temp = n.clone();
    for p in prime_factors {
        let mut p_powers = Vec::new();
        let mut pk = BigUint::one();
        while (&temp % p).is_zero() {
            temp /= p;
            pk *= p;
            p_powers.push(pk.clone());
        }
        let prev = divs.clone();
        for pp in &p_powers {
            for d in &prev {
                divs.push(d * pp);
            }
        }
    }
    divs.sort();
    divs
}

/// Count elliptic points of order 2 for Gamma_0(N). BigUint version.
/// Uses wheel factorization (mod 6) for trial division.
pub fn count_elliptic_2_big(n: &BigUint) -> BigUint {
    let four = BigUint::from(4u64);
    if (n % &four).is_zero() {
        return BigUint::zero();
    }
    let factors = trial_factor_big(n);
    count_elliptic_from_factors(n, &factors, -1)
}

/// Count elliptic points of order 3 for Gamma_0(N). BigUint version.
/// Uses wheel factorization (mod 6) for trial division.
pub fn count_elliptic_3_big(n: &BigUint) -> BigUint {
    let nine = BigUint::from(9u64);
    if (n % &nine).is_zero() {
        return BigUint::zero();
    }
    let factors = trial_factor_big(n);
    count_elliptic_from_factors(n, &factors, -3)
}

/// Trial-factor N using wheel mod 6. Returns the list of distinct prime factors.
fn trial_factor_big(n: &BigUint) -> Vec<BigUint> {
    let mut factors = Vec::new();
    let mut temp = n.clone();
    let two = BigUint::from(2u64);
    let three = BigUint::from(3u64);

    if (&temp % &two).is_zero() {
        factors.push(two.clone());
        while (&temp % &two).is_zero() {
            temp /= &two;
        }
    }
    if (&temp % &three).is_zero() {
        factors.push(three.clone());
        while (&temp % &three).is_zero() {
            temp /= &three;
        }
    }

    let mut p = 5u64;
    while BigUint::from(p) * BigUint::from(p) <= temp {
        let p_big = BigUint::from(p);
        if (&temp % &p_big).is_zero() {
            factors.push(p_big.clone());
            while (&temp % &p_big).is_zero() {
                temp /= &p_big;
            }
        }
        let p2 = p + 2;
        let p2_big = BigUint::from(p2);
        if (&temp % &p2_big).is_zero() {
            factors.push(p2_big.clone());
            while (&temp % &p2_big).is_zero() {
                temp /= &p2_big;
            }
        }
        p += 6;
    }
    if temp > BigUint::one() {
        factors.push(temp);
    }
    factors
}

/// Count cusps of Gamma_0(N). BigUint version.
/// c(N) = sum_{d|N} phi(gcd(d, N/d))
pub fn count_cusps_big(n: &BigUint) -> BigUint {
    let factors = trial_factor_big(n);
    count_cusps_from_factors(n, &factors)
}

/// Euler's totient function. BigUint version.
/// Uses wheel factorization (mod 6) for speed.
pub fn euler_phi_big(n: &BigUint) -> BigUint {
    if *n <= BigUint::one() {
        return n.clone();
    }
    let mut result = n.clone();
    let mut temp = n.clone();
    let two = BigUint::from(2u64);
    let three = BigUint::from(3u64);

    if (&temp % &two).is_zero() {
        result -= &result / &two;
        while (&temp % &two).is_zero() {
            temp /= &two;
        }
    }
    if (&temp % &three).is_zero() {
        result -= &result / &three;
        while (&temp % &three).is_zero() {
            temp /= &three;
        }
    }

    let mut p = 5u64;
    while BigUint::from(p) * BigUint::from(p) <= temp {
        let p_big = BigUint::from(p);
        if (&temp % &p_big).is_zero() {
            result -= &result / &p_big;
            while (&temp % &p_big).is_zero() {
                temp /= &p_big;
            }
        }
        let p2 = p + 2;
        let p2_big = BigUint::from(p2);
        if (&temp % &p2_big).is_zero() {
            result -= &result / &p2_big;
            while (&temp % &p2_big).is_zero() {
                temp /= &p2_big;
            }
        }
        p += 6;
    }
    if temp > BigUint::one() {
        result -= &result / &temp;
    }
    result
}

/// GCD for BigUint.
pub fn gcd_big(a: &BigUint, b: &BigUint) -> BigUint {
    a.gcd(b)
}

/// Get all divisors of n. BigUint version.
/// Uses trial factoring with wheel, then builds divisors from prime factorization.
pub fn get_divisors_big(n: &BigUint) -> Vec<BigUint> {
    let factors = trial_factor_big(n);
    divisors_from_prime_factors(n, &factors)
}

/// Kronecker/Jacobi symbol (a/n) for BigInt a and BigUint n.
/// Returns -1, 0, or 1.
pub fn kronecker_symbol_big(a: &BigInt, n: &BigUint) -> i64 {
    let zero_u = BigUint::zero();
    let one_u = BigUint::one();
    let two_u = BigUint::from(2u64);

    if *n == zero_u {
        let abs_a = a.magnitude();
        return if *abs_a == one_u { 1 } else { 0 };
    }
    if *n == one_u {
        return 1;
    }
    if *n == two_u {
        let a_mod_2 = a.mod_floor(&BigInt::from(2i64));
        if a_mod_2.is_zero() {
            return 0;
        }
        let eight = BigInt::from(8i64);
        let r = a.mod_floor(&eight);
        let r_u64 = r.to_u64().unwrap_or(0);
        return if r_u64 == 1 || r_u64 == 7 { 1 } else { -1 };
    }

    // For odd n, use Jacobi symbol computation
    let n_int = BigInt::from(n.clone());
    let a_mod = a.mod_floor(&n_int);
    let mut a_val = a_mod.to_biguint().unwrap_or_else(|| BigUint::zero());
    let mut n_val = n.clone();
    let mut result: i64 = 1;

    while !a_val.is_zero() {
        while (&a_val % &two_u).is_zero() {
            a_val /= &two_u;
            let r = &n_val % BigUint::from(8u64);
            let r_u64 = r.to_u64().unwrap_or(0);
            if r_u64 == 3 || r_u64 == 5 {
                result = -result;
            }
        }
        std::mem::swap(&mut a_val, &mut n_val);
        let four = BigUint::from(4u64);
        let a_mod4 = &a_val % &four;
        let n_mod4 = &n_val % &four;
        let three = BigUint::from(3u64);
        if a_mod4 == three && n_mod4 == three {
            result = -result;
        }
        a_val %= &n_val;
    }
    if n_val == one_u {
        result
    } else {
        0
    }
}

/// Dim of new subspace S_2^new(Gamma_0(N)). BigUint version.
pub fn dim_s2_new_big(n: &BigUint) -> BigUint {
    let total = dim_s2_big(n);
    let divs = get_divisors_big(n);
    let mut old_dim = BigUint::zero();
    for d in &divs {
        if d == n {
            continue;
        }
        let nd = n / d;
        let num_embeddings = BigUint::from(get_divisors_big(&nd).len() as u64);
        old_dim += &num_embeddings * dim_s2_new_inner_big(d);
    }
    if total >= old_dim {
        total - old_dim
    } else {
        BigUint::zero()
    }
}

fn dim_s2_new_inner_big(n: &BigUint) -> BigUint {
    if *n <= BigUint::one() || is_prime_big(n) {
        return dim_s2_big(n);
    }
    let total = dim_s2_big(n);
    let divs = get_divisors_big(n);
    let mut old_dim = BigUint::zero();
    for d in &divs {
        if d == n {
            continue;
        }
        let nd = n / d;
        let num_embeddings = BigUint::from(get_divisors_big(&nd).len() as u64);
        old_dim += &num_embeddings * dim_s2_new_inner_big(d);
    }
    if total >= old_dim {
        total - old_dim
    } else {
        BigUint::zero()
    }
}

/// Miller-Rabin primality test for BigUint. Deterministic for values up to 3,317,044,064,679,887,385,961,981.
pub fn is_prime_big(n: &BigUint) -> bool {
    let one = BigUint::one();
    let two = BigUint::from(2u64);
    let three = BigUint::from(3u64);

    if *n < two {
        return false;
    }
    if *n == two || *n == three {
        return true;
    }
    if (n % &two).is_zero() || (n % &three).is_zero() {
        return false;
    }

    // Write n-1 = 2^r * d
    let n_minus_1 = n - &one;
    let mut d = n_minus_1.clone();
    let mut r = 0u64;
    while (&d % &two).is_zero() {
        d /= &two;
        r += 1;
    }

    // Deterministic witnesses sufficient for numbers up to 64 bits
    let witnesses: Vec<BigUint> = vec![2u64, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        .into_iter()
        .map(BigUint::from)
        .collect();

    'witness: for a in &witnesses {
        if a >= n {
            continue;
        }
        let mut x = mod_pow_big(a, &d, n);
        if x == one || x == n_minus_1 {
            continue;
        }
        for _ in 0..r - 1 {
            x = mod_pow_big(&x, &two, n);
            if x == n_minus_1 {
                continue 'witness;
            }
        }
        return false;
    }
    true
}

/// Modular exponentiation: base^exp mod modulus. BigUint version.
fn mod_pow_big(base: &BigUint, exp: &BigUint, modulus: &BigUint) -> BigUint {
    if modulus.is_one() {
        return BigUint::zero();
    }
    let mut result = BigUint::one();
    let mut base = base % modulus;
    let mut exp = exp.clone();
    let two = BigUint::from(2u64);

    while !exp.is_zero() {
        if (&exp % &two) == BigUint::one() {
            result = (&result * &base) % modulus;
        }
        exp /= &two;
        base = (&base * &base) % modulus;
    }
    result
}

/// Integer square root for BigUint. Returns floor(sqrt(n)).
pub fn isqrt_big(n: &BigUint) -> BigUint {
    if n.is_zero() {
        return BigUint::zero();
    }
    if *n == BigUint::one() {
        return BigUint::one();
    }

    // Newton's method: x_{n+1} = (x_n + n/x_n) / 2
    let two = BigUint::from(2u64);
    let mut x = n.clone();
    let mut y = (&x + BigUint::one()) / &two;
    while y < x {
        x = y.clone();
        y = (&x + n / &x) / &two;
    }
    x
}

// ============================================================
// Original u64 versions (kept for backward compatibility)
// ============================================================

/// Dimension of S_2(Gamma_0(N)) for any N.
/// Uses the formula involving genus, cusps, and elliptic points.
pub fn dim_s2(n: u64) -> u64 {
    if n <= 1 {
        return 0;
    }

    // Genus formula for Gamma_0(N):
    // dim S_2 = 1 + mu/12 - nu_2/4 - nu_3/3 - c/2
    // where:
    // - mu = psi(N) = index [SL2(Z) : Gamma_0(N)]
    // - nu_2 = number of elliptic points of order 2
    // - nu_3 = number of elliptic points of order 3
    // - c = number of cusps
    // For weight 2: dim S_2 = genus

    let mu = psi(n); // Index [SL2(Z) : Gamma_0(N)]
    let nu2 = count_elliptic_2(n);
    let nu3 = count_elliptic_3(n);
    let cusps = count_cusps(n);

    // genus = 1 + mu/12 - nu2/4 - nu3/3 - cusps/2
    // dim S_2 = genus (for weight 2, k >= 2)
    let val = 1.0 + (mu as f64) / 12.0 - (nu2 as f64) / 4.0 - (nu3 as f64) / 3.0
        - (cusps as f64) / 2.0;
    if val < 0.0 {
        0
    } else {
        val.round() as u64
    }
}

/// Psi(N) = N * prod_{p|N} (1 + 1/p) -- the index of Gamma_0(N) in SL_2(Z).
fn psi(n: u64) -> u64 {
    let mut result = n;
    let mut temp = n;
    let mut p = 2u64;
    let mut factors = Vec::new();
    while p * p <= temp {
        if temp % p == 0 {
            factors.push(p);
            while temp % p == 0 {
                temp /= p;
            }
        }
        p += 1;
    }
    if temp > 1 {
        factors.push(temp);
    }

    // result = N * prod_{p|N} (1 + 1/p)
    // = N * prod_{p|N} (p+1)/p
    // Do integer arithmetic: multiply by (p+1), divide by p
    for p in &factors {
        result = result / p * (p + 1);
    }
    result
}

/// Count elliptic points of order 2 for Gamma_0(N).
fn count_elliptic_2(n: u64) -> u64 {
    if n % 4 == 0 {
        return 0;
    }
    // Product over p|N of (1 + (-1/p))
    let mut result = 1i64;
    let mut temp = n;
    let mut p = 2u64;
    while p * p <= temp {
        if temp % p == 0 {
            while temp % p == 0 {
                temp /= p;
            }
            let legendre = if p == 2 {
                0
            } else {
                kronecker_symbol(-1, p)
            };
            result *= 1 + legendre;
        }
        p += 1;
    }
    if temp > 1 {
        let legendre = kronecker_symbol(-1, temp);
        result *= 1 + legendre;
    }
    if result < 0 {
        0
    } else {
        result as u64
    }
}

/// Count elliptic points of order 3 for Gamma_0(N).
fn count_elliptic_3(n: u64) -> u64 {
    if n % 9 == 0 {
        return 0;
    }
    let mut result = 1i64;
    let mut temp = n;
    let mut p = 2u64;
    while p * p <= temp {
        if temp % p == 0 {
            while temp % p == 0 {
                temp /= p;
            }
            let legendre = if p == 3 {
                0
            } else {
                kronecker_symbol(-3, p)
            };
            result *= 1 + legendre;
        }
        p += 1;
    }
    if temp > 1 {
        let legendre = kronecker_symbol(-3, temp);
        result *= 1 + legendre;
    }
    if result < 0 {
        0
    } else {
        result as u64
    }
}

/// Count cusps of Gamma_0(N).
fn count_cusps(n: u64) -> u64 {
    // c(N) = sum_{d|N} phi(gcd(d, N/d))
    let divisors = get_divisors(n);
    let mut count = 0u64;
    for &d in &divisors {
        let nd = n / d;
        count += euler_phi(gcd_u64(d, nd));
    }
    count
}

fn euler_phi(n: u64) -> u64 {
    if n <= 1 {
        return n;
    }
    let mut result = n;
    let mut temp = n;
    let mut p = 2u64;
    while p * p <= temp {
        if temp % p == 0 {
            while temp % p == 0 {
                temp /= p;
            }
            result -= result / p;
        }
        p += 1;
    }
    if temp > 1 {
        result -= result / temp;
    }
    result
}

fn gcd_u64(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

fn get_divisors(n: u64) -> Vec<u64> {
    let mut divs = Vec::new();
    let mut i = 1;
    while i * i <= n {
        if n % i == 0 {
            divs.push(i);
            if i != n / i {
                divs.push(n / i);
            }
        }
        i += 1;
    }
    divs.sort();
    divs
}

fn kronecker_symbol(a: i64, n: u64) -> i64 {
    // Standard Jacobi/Kronecker symbol
    if n == 0 {
        return if a.unsigned_abs() == 1 { 1 } else { 0 };
    }
    if n == 1 {
        return 1;
    }
    if n == 2 {
        if a % 2 == 0 {
            return 0;
        }
        let r = ((a % 8) + 8) % 8;
        return if r == 1 || r == 7 { 1 } else { -1 };
    }
    // For odd n, use Jacobi
    let mut a = ((a % n as i64) + n as i64) as u64 % n;
    let mut n = n;
    let mut result = 1i64;
    while a != 0 {
        while a % 2 == 0 {
            a /= 2;
            let r = n % 8;
            if r == 3 || r == 5 {
                result = -result;
            }
        }
        std::mem::swap(&mut a, &mut n);
        if a % 4 == 3 && n % 4 == 3 {
            result = -result;
        }
        a %= n;
    }
    if n == 1 {
        result
    } else {
        0
    }
}

/// Dim of new subspace S_2^new(Gamma_0(N)).
pub fn dim_s2_new(n: u64) -> u64 {
    let total = dim_s2(n);
    let divs = get_divisors(n);
    let mut old_dim = 0u64;
    for &d in &divs {
        if d == n {
            continue;
        }
        let nd = n / d;
        let num_embeddings = get_divisors(nd).len() as u64;
        old_dim += num_embeddings * dim_s2_new_inner(d);
    }
    if total >= old_dim {
        total - old_dim
    } else {
        0
    }
}

fn dim_s2_new_inner(n: u64) -> u64 {
    // For prime n, dim_new = dim_total
    // For general n, recursive
    if is_prime(n) || n <= 1 {
        return dim_s2(n);
    }
    let total = dim_s2(n);
    let divs = get_divisors(n);
    let mut old_dim = 0u64;
    for &d in &divs {
        if d == n {
            continue;
        }
        let nd = n / d;
        let num_embeddings = get_divisors(nd).len() as u64;
        old_dim += num_embeddings * dim_s2_new_inner(d);
    }
    if total >= old_dim {
        total - old_dim
    } else {
        0
    }
}

fn is_prime(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    if n < 4 {
        return true;
    }
    if n % 2 == 0 || n % 3 == 0 {
        return false;
    }
    let mut i = 5;
    while i * i <= n {
        if n % i == 0 || n % (i + 2) == 0 {
            return false;
        }
        i += 6;
    }
    true
}

/// Compute Ramanujan-Petersson bound on tr(T_l) at level N.
/// For each newform f of level d|N, |a_l(f)| <= 2*sqrt(l).
/// The total trace is bounded by:
/// |tr(T_l)| <= dim_S_2(N) * 2 * sqrt(l)
pub fn trace_bound(n: u64, l: u64) -> f64 {
    let dim = dim_s2(n);
    (dim as f64) * 2.0 * (l as f64).sqrt()
}

/// Compute the old-new structure constraints.
/// For N = pq with p, q prime:
/// - dim S_2^old(N) = 2*dim(p) + 2*dim(q)
/// - dim S_2^new(N) = dim(N) - dim_old
///
/// The trace decomposes as:
/// tr(T_l, level N) = 2*tr(T_l, level p) + 2*tr(T_l, level q) + tr(T_l, new at N)
///
/// This gives us: for each l, tr(T_l, N) constrains the SUM
/// 2*tr(T_l, p) + 2*tr(T_l, q) + tr(new)
/// where |tr(new)| <= dim_new * 2*sqrt(l)
#[derive(Debug, Clone)]
pub struct TraceConstraint {
    pub l: u64,
    pub total_trace_bound: f64,
    pub new_trace_bound: f64,
    pub dim_total: u64,
    pub dim_new: u64,
    pub dim_old: u64,
}

pub fn compute_trace_constraints(n: u64, primes_l: &[u64]) -> Vec<TraceConstraint> {
    let dim_total = dim_s2(n);
    let dim_new = dim_s2_new(n);
    let dim_old = dim_total - dim_new;

    primes_l
        .iter()
        .filter(|&&l| n % l != 0)
        .map(|&l| TraceConstraint {
            l,
            total_trace_bound: trace_bound(n, l),
            new_trace_bound: (dim_new as f64) * 2.0 * (l as f64).sqrt(),
            dim_total,
            dim_new,
            dim_old,
        })
        .collect()
}

/// For a KNOWN factorization N = pq, compute the actual trace structure.
/// This is the "oracle" version for testing -- in practice we don't know p, q.
#[derive(Debug, Clone)]
pub struct OracleTraceData {
    pub n: u64,
    pub p: u64,
    pub q: u64,
    pub l: u64,
    pub dim_p: u64,
    pub dim_q: u64,
    pub dim_n: u64,
    pub dim_new_n: u64,
    pub trace_bound_old: f64,
}

pub fn oracle_trace_data(p: u64, q: u64, l: u64) -> OracleTraceData {
    let n = p * q;
    OracleTraceData {
        n,
        p,
        q,
        l,
        dim_p: dim_s2(p),
        dim_q: dim_s2(q),
        dim_n: dim_s2(n),
        dim_new_n: dim_s2_new(n),
        trace_bound_old: 2.0 * (dim_s2(p) + dim_s2(q)) as f64 * 2.0 * (l as f64).sqrt(),
    }
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dim_s2_big_matches_u64() {
        // Verify BigUint version matches u64 for known values
        let known_dims: Vec<(u64, u64)> = vec![
            (11, 1), (23, 2), (29, 2), (37, 2), (41, 3),
            (43, 3), (53, 4), (61, 4), (67, 5), (71, 6), (77, 7),
        ];
        for (n, expected) in &known_dims {
            let big_n = BigUint::from(*n);
            let big_result = dim_s2_big(&big_n);
            let u64_result = dim_s2(*n);
            assert_eq!(
                big_result,
                BigUint::from(*expected),
                "dim_s2_big({}) = {} expected {}",
                n, big_result, expected
            );
            assert_eq!(
                u64_result, *expected,
                "dim_s2({}) = {} expected {}",
                n, u64_result, expected
            );
        }
    }

    #[test]
    fn test_dim_s2_new_big_matches_u64() {
        let test_values = [11u64, 23, 77, 143, 221, 323];
        for &n in &test_values {
            let big_n = BigUint::from(n);
            let big_result = dim_s2_new_big(&big_n);
            let u64_result = dim_s2_new(n);
            assert_eq!(
                big_result,
                BigUint::from(u64_result),
                "dim_s2_new mismatch at n={}: big={}, u64={}",
                n, big_result, u64_result
            );
        }
    }

    #[test]
    fn test_psi_big_matches_u64() {
        for n in [6u64, 10, 12, 30, 77, 143, 221, 1000] {
            let big_n = BigUint::from(n);
            let big_result = psi_big(&big_n);
            let u64_result = psi(n);
            assert_eq!(
                big_result,
                BigUint::from(u64_result),
                "psi mismatch at n={}: big={}, u64={}",
                n, big_result, u64_result
            );
        }
    }

    #[test]
    fn test_kronecker_big_matches_u64() {
        for n in [3u64, 5, 7, 11, 13, 17, 19, 23, 29, 31] {
            let big_n = BigUint::from(n);
            for a in [-3i64, -1, 0, 1, 2, 3, 5] {
                let big_a = BigInt::from(a);
                let big_result = kronecker_symbol_big(&big_a, &big_n);
                let u64_result = kronecker_symbol(a, n);
                assert_eq!(
                    big_result, u64_result,
                    "kronecker({}, {}) mismatch: big={}, u64={}",
                    a, n, big_result, u64_result
                );
            }
        }
    }

    #[test]
    fn test_is_prime_big() {
        let primes = [2u64, 3, 5, 7, 11, 13, 97, 1009, 104729];
        for &p in &primes {
            assert!(
                is_prime_big(&BigUint::from(p)),
                "{} should be prime",
                p
            );
        }
        let composites = [4u64, 6, 9, 15, 77, 143, 1001];
        for &c in &composites {
            assert!(
                !is_prime_big(&BigUint::from(c)),
                "{} should be composite",
                c
            );
        }
    }

    #[test]
    fn test_isqrt_big() {
        assert_eq!(isqrt_big(&BigUint::from(0u64)), BigUint::zero());
        assert_eq!(isqrt_big(&BigUint::from(1u64)), BigUint::one());
        assert_eq!(isqrt_big(&BigUint::from(4u64)), BigUint::from(2u64));
        assert_eq!(isqrt_big(&BigUint::from(9u64)), BigUint::from(3u64));
        assert_eq!(isqrt_big(&BigUint::from(10u64)), BigUint::from(3u64));
        assert_eq!(isqrt_big(&BigUint::from(100u64)), BigUint::from(10u64));
        // Large value
        let big = BigUint::from(10000000000u64); // 10^10
        let root = isqrt_big(&big);
        assert_eq!(root, BigUint::from(100000u64)); // 10^5
    }

    #[test]
    fn test_euler_phi_big_matches_u64() {
        for n in [1u64, 2, 6, 10, 12, 30, 77, 100] {
            let big_result = euler_phi_big(&BigUint::from(n));
            let u64_result = euler_phi(n);
            assert_eq!(
                big_result,
                BigUint::from(u64_result),
                "euler_phi mismatch at n={}",
                n
            );
        }
    }

    #[test]
    fn test_count_cusps_big_matches_u64() {
        for n in [2u64, 6, 11, 30, 77, 143] {
            let big_result = count_cusps_big(&BigUint::from(n));
            let u64_result = count_cusps(n);
            assert_eq!(
                big_result,
                BigUint::from(u64_result),
                "count_cusps mismatch at n={}",
                n
            );
        }
    }

    #[test]
    fn test_dim_s2_big_semiprimes() {
        // Test on some semiprimes that u64 can also handle
        let cases: [(u64, u64, u64); 4] = [(77, 7, 11), (143, 11, 13), (221, 13, 17), (323, 17, 19)];
        for (n, p, q) in &cases {
            let big_n = BigUint::from(*n);
            let big_p = BigUint::from(*p);
            let big_q = BigUint::from(*q);

            let dim_n = dim_s2_big(&big_n);
            let dim_p = dim_s2_big(&big_p);
            let dim_q = dim_s2_big(&big_q);
            let dim_new = dim_s2_new_big(&big_n);

            // dim_old = dim_total - dim_new
            let dim_old = &dim_n - &dim_new;
            // For N = pq with p,q prime: dim_old = 2*dim(p) + 2*dim(q)
            let expected_old = BigUint::from(2u64) * &dim_p + BigUint::from(2u64) * &dim_q;
            assert_eq!(
                dim_old, expected_old,
                "old-new decomposition mismatch for {} = {} x {}: dim_old={}, expected={}",
                n, p, q, dim_old, expected_old
            );
        }
    }
}
