use rug::Integer;

/// Integer square root: floor(sqrt(n)).
pub fn isqrt(n: &Integer) -> Integer {
    if *n <= 0 {
        return Integer::from(0);
    }
    n.clone().sqrt()
}

/// Integer k-th root: floor(n^(1/k)).
pub fn nth_root(n: &Integer, k: u32) -> Integer {
    if *n <= 0 || k == 0 {
        return Integer::from(0);
    }
    if k == 1 {
        return n.clone();
    }
    n.clone().root(k)
}

/// Sieve of Eratosthenes: return all primes up to bound (inclusive).
pub fn sieve_primes(bound: u64) -> Vec<u64> {
    if bound < 2 {
        return vec![];
    }
    let limit = bound as usize;
    let mut is_prime = vec![true; limit + 1];
    is_prime[0] = false;
    if limit >= 1 {
        is_prime[1] = false;
    }
    let mut i = 2;
    while i * i <= limit {
        if is_prime[i] {
            let mut j = i * i;
            while j <= limit {
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
        .collect()
}

/// Modular inverse of a mod m using extended GCD. Returns None if gcd(a, m) != 1.
pub fn mod_inverse(a: u64, m: u64) -> Option<u64> {
    let a = Integer::from(a);
    let m_int = Integer::from(m);
    match a.invert(&m_int) {
        Ok(inv) => inv.to_u64(),
        Err(_) => None,
    }
}

/// Native u64 modular inverse using extended Euclidean algorithm.
/// Returns x such that a*x ≡ 1 (mod m), or None if gcd(a,m) != 1.
/// Uses i128 arithmetic to avoid overflow for large inputs.
/// Preferred over `mod_inverse` in hot loops (avoids rug::Integer allocation).
pub fn mod_inverse_u64(a: u64, m: u64) -> Option<u64> {
    if m <= 1 {
        return None;
    }
    let (mut old_r, mut r) = (a as i128, m as i128);
    let (mut old_s, mut s) = (1i128, 0i128);

    while r != 0 {
        let q = old_r / r;
        let temp_r = r;
        r = old_r - q * r;
        old_r = temp_r;
        let temp_s = s;
        s = old_s - q * s;
        old_s = temp_s;
    }

    if old_r != 1 {
        return None;
    }
    Some(old_s.rem_euclid(m as i128) as u64)
}

/// Tonelli-Shanks: find r such that r^2 ≡ n (mod p). Returns None if n is not a QR mod p.
pub fn tonelli_shanks(n: u64, p: u64) -> Option<u64> {
    if p == 2 {
        return Some(n % 2);
    }
    let n_int = Integer::from(n);
    let p_int = Integer::from(p);

    // Check if n is a quadratic residue via Euler's criterion
    let exp = Integer::from((p - 1) / 2);
    let legendre = n_int.clone().pow_mod(&exp, &p_int).unwrap();
    if legendre != 1 {
        return None;
    }

    // Factor out powers of 2 from p-1: p-1 = Q * 2^S
    let mut q = p - 1;
    let mut s: u32 = 0;
    while q % 2 == 0 {
        q /= 2;
        s += 1;
    }

    if s == 1 {
        // p ≡ 3 (mod 4): simple case
        let exp = Integer::from((p + 1) / 4);
        let r = n_int.pow_mod(&exp, &p_int).unwrap();
        return r.to_u64();
    }

    // Find a quadratic non-residue z
    let mut z = 2u64;
    loop {
        let z_int = Integer::from(z);
        let exp = Integer::from((p - 1) / 2);
        let l = z_int.pow_mod(&exp, &p_int).unwrap();
        if l == p - 1 {
            break;
        }
        z += 1;
    }

    let mut m_val = s;
    let mut c = Integer::from(z).pow_mod(&Integer::from(q), &p_int).unwrap();
    let mut t = n_int.clone().pow_mod(&Integer::from(q), &p_int).unwrap();
    let mut r = n_int.pow_mod(&Integer::from((q + 1) / 2), &p_int).unwrap();

    loop {
        if t == 0 {
            return Some(0);
        }
        if t == 1 {
            return r.to_u64();
        }

        // Find the least i such that t^(2^i) ≡ 1 (mod p)
        let mut i = 0u32;
        let mut temp = t.clone();
        while temp != 1 {
            temp = temp.clone().pow_mod(&Integer::from(2), &p_int).unwrap();
            i += 1;
            if i == m_val {
                return None;
            }
        }

        let exp = Integer::from(1u64 << (m_val - i - 1));
        let b = c.clone().pow_mod(&exp, &p_int).unwrap();
        m_val = i;
        c = b.clone().pow_mod(&Integer::from(2), &p_int).unwrap();
        t = (t * &c) % &p_int;
        r = (r * &b) % &p_int;
    }
}

/// Evaluate polynomial with i64 coefficients at x, mod m.
/// Coefficients are [c0, c1, ..., cd] (low-degree first).
pub fn eval_poly_mod(coeffs: &[i64], x: u64, m: u64) -> u64 {
    if coeffs.is_empty() {
        return 0;
    }
    let m_i = m as i128;
    let x_i = x as i128;
    let mut result: i128 = 0;
    for &c in coeffs.iter().rev() {
        result = (result * x_i + c as i128).rem_euclid(m_i);
    }
    result as u64
}

/// Find all roots of polynomial f(x) ≡ 0 (mod p) by brute force.
/// Coefficients are [c0, c1, ..., cd] (low-degree first).
/// For M1, brute force is fine since factor base primes are small.
pub fn find_polynomial_roots_mod_p(coeffs: &[i64], p: u64) -> Vec<u64> {
    let mut roots = Vec::new();
    for x in 0..p {
        if eval_poly_mod(coeffs, x, p) == 0 {
            roots.push(x);
        }
    }
    roots
}

/// Modular inverse of a mod m for rug::Integer. Returns None if not invertible.
pub fn mod_inverse_int(a: &Integer, m: &Integer) -> Option<Integer> {
    match a.clone().invert(m) {
        Ok(inv) => Some(inv),
        Err(_) => None,
    }
}

/// Tonelli-Shanks for rug::Integer: find r such that r² ≡ n (mod p).
pub fn tonelli_shanks_int(n: &Integer, p: &Integer) -> Option<Integer> {
    let n_mod = Integer::from(n % p);
    if n_mod == 0 {
        return Some(Integer::from(0));
    }
    if *p == 2 {
        return Some(n_mod);
    }

    // Euler's criterion
    let exp = Integer::from(p - 1) / 2;
    let legendre = n_mod.clone().pow_mod(&exp, p).unwrap();
    if legendre != 1 {
        return None;
    }

    // Factor p-1 = q * 2^s
    let mut q = Integer::from(p - 1);
    let mut s: u32 = 0;
    while q.is_even() {
        q /= 2;
        s += 1;
    }

    if s == 1 {
        let exp = Integer::from(p + 1) / 4;
        let r = n_mod.pow_mod(&exp, p).unwrap();
        return Some(r);
    }

    // Find quadratic non-residue
    let mut z = Integer::from(2);
    loop {
        let l = z.clone().pow_mod(&exp, p).unwrap();
        if l == Integer::from(p - 1) {
            break;
        }
        z += 1;
    }

    let mut m_val = s;
    let mut c = z.pow_mod(&q, p).unwrap();
    let mut t = n_mod.clone().pow_mod(&q, p).unwrap();
    let q_plus_1_half = Integer::from(&q + 1) / 2;
    let mut r = n_mod.pow_mod(&q_plus_1_half, p).unwrap();

    loop {
        if t == 0 {
            return Some(Integer::from(0));
        }
        if t == 1 {
            return Some(r);
        }

        let mut i = 0u32;
        let mut temp = t.clone();
        let two = Integer::from(2);
        while temp != 1 {
            temp = temp.pow_mod(&two, p).unwrap();
            i += 1;
            if i == m_val {
                return None;
            }
        }

        let exp = Integer::from(1u64) << (m_val - i - 1);
        let b = c.clone().pow_mod(&exp, p).unwrap();
        m_val = i;
        c = Integer::from(&b * &b) % p;
        t = Integer::from(&t * &c) % p;
        r = Integer::from(&r * &b) % p;
    }
}

/// Evaluate polynomial with Integer coefficients at an Integer point, mod modulus.
/// Coefficients are [c0, c1, ..., cd] (low-degree first). Uses Horner's method.
pub fn eval_poly_int(coeffs: &[Integer], x: &Integer, modulus: &Integer) -> Integer {
    let mut result = Integer::from(0);
    for c in coeffs.iter().rev() {
        result = Integer::from(Integer::from(&result * x) + c) % modulus;
    }
    if result < 0 {
        result += modulus;
    }
    result
}

/// Evaluate polynomial derivative at an Integer point, mod modulus.
/// f'(x) = c1 + 2*c2*x + 3*c3*x² + ...
pub fn eval_poly_deriv_int(coeffs: &[Integer], x: &Integer, modulus: &Integer) -> Integer {
    if coeffs.len() <= 1 {
        return Integer::from(0);
    }
    let mut result = Integer::from(0);
    for (i, c) in coeffs.iter().enumerate().skip(1).rev() {
        result = Integer::from(Integer::from(&result * x) + Integer::from(c * i as u64)) % modulus;
    }
    if result < 0 {
        result += modulus;
    }
    result
}

/// Lagrange interpolation mod modulus.
/// Given points (xs[i], ys[i]), find polynomial of degree < d evaluated at result coefficients.
/// Returns polynomial coefficients [c0, c1, ..., c_{d-1}].
pub fn lagrange_interpolation_mod(
    xs: &[Integer],
    ys: &[Integer],
    modulus: &Integer,
) -> Vec<Integer> {
    let d = xs.len();

    // Build the polynomial using the coefficient form
    // L_i(x) = y_i * prod_{j != i} (x - x_j) / (x_i - x_j)
    // We accumulate coefficients directly.

    let mut result = vec![Integer::from(0); d];

    for i in 0..d {
        // Compute denominator: prod_{j != i} (xs[i] - xs[j]) mod modulus
        let mut denom = Integer::from(1);
        for j in 0..d {
            if j != i {
                let diff = Integer::from(&xs[i] - &xs[j]) % modulus;
                let diff = if diff < 0 { diff + modulus } else { diff };
                denom = Integer::from(&denom * &diff) % modulus;
            }
        }
        let denom_inv = match mod_inverse_int(&denom, modulus) {
            Some(inv) => inv,
            None => return vec![Integer::from(0); d], // shouldn't happen
        };
        let coeff = Integer::from(&ys[i] * &denom_inv) % modulus;

        // Compute L_i(x) = prod_{j != i} (x - x_j) as polynomial coefficients
        // Start with [1] and multiply by (x - x_j) for each j != i
        let mut basis = vec![Integer::from(0); d];
        basis[0] = Integer::from(1);
        let mut deg = 0;

        for j in 0..d {
            if j == i {
                continue;
            }
            let neg_xj = Integer::from(-&xs[j]) % modulus;
            let neg_xj = if neg_xj < 0 { neg_xj + modulus } else { neg_xj };

            // Multiply basis polynomial by (x - x_j)
            // New coefficients: new[k] = basis[k-1] + (-x_j) * basis[k]
            let mut new_basis = vec![Integer::from(0); d];
            for k in (0..=deg + 1).rev() {
                let from_shift = if k > 0 {
                    &basis[k - 1]
                } else {
                    &Integer::from(0)
                };
                let from_scale = Integer::from(&basis[k] * &neg_xj) % modulus;
                new_basis[k] = Integer::from(from_shift + &from_scale) % modulus;
                if new_basis[k] < 0 {
                    new_basis[k] += modulus;
                }
            }
            basis = new_basis;
            deg += 1;
        }

        // Add coeff * basis to result
        for k in 0..d {
            result[k] = Integer::from(&result[k] + Integer::from(&coeff * &basis[k])) % modulus;
            if result[k] < 0 {
                result[k] += modulus;
            }
        }
    }

    result
}

// ---------------------------------------------------------------------------
// Polynomial arithmetic over F_p (for irreducibility testing)
// ---------------------------------------------------------------------------

/// Polynomial over F_p stored in ascending degree order: poly[i] = coefficient of x^i.
/// The zero polynomial is represented as an empty Vec.
type PolyFp = Vec<u64>;

/// Strip trailing zeros from a polynomial.
fn poly_fp_normalize(a: &mut PolyFp) {
    while a.last() == Some(&0) {
        a.pop();
    }
}

/// Polynomial multiplication mod p.
fn poly_fp_mul(a: &PolyFp, b: &PolyFp, p: u64) -> PolyFp {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    let mut c = vec![0u64; a.len() + b.len() - 1];
    for (i, &ai) in a.iter().enumerate() {
        if ai == 0 {
            continue;
        }
        for (j, &bj) in b.iter().enumerate() {
            c[i + j] = ((c[i + j] as u128 + ai as u128 * bj as u128) % p as u128) as u64;
        }
    }
    poly_fp_normalize(&mut c);
    c
}

/// Polynomial remainder: a mod b over F_p.
fn poly_fp_rem(a: &PolyFp, b: &PolyFp, p: u64) -> PolyFp {
    if b.is_empty() {
        return a.clone();
    }
    let mut r = a.clone();
    let db = b.len() - 1;
    let lead_b_inv = mod_inverse(*b.last().unwrap(), p).unwrap();
    while r.len() > db {
        let coeff =
            ((r.last().copied().unwrap_or(0) as u128 * lead_b_inv as u128) % p as u128) as u64;
        let shift = r.len() - 1 - db;
        for (i, &bi) in b.iter().enumerate() {
            let sub = (coeff as u128 * bi as u128) % p as u128;
            r[shift + i] = ((r[shift + i] as u128 + p as u128 - sub) % p as u128) as u64;
        }
        poly_fp_normalize(&mut r);
    }
    r
}

/// Polynomial GCD over F_p, returned as monic.
fn poly_fp_gcd(a: &PolyFp, b: &PolyFp, p: u64) -> PolyFp {
    let mut a = a.clone();
    let mut b = b.clone();
    while !b.is_empty() {
        let r = poly_fp_rem(&a, &b, p);
        a = b;
        b = r;
    }
    // Make monic
    if let Some(&lead) = a.last() {
        if lead != 0 && lead != 1 {
            let inv = mod_inverse(lead, p).unwrap();
            for c in a.iter_mut() {
                *c = (*c as u128 * inv as u128 % p as u128) as u64;
            }
        }
    }
    a
}

/// Compute base^exp mod modulus_poly over F_p using repeated squaring.
fn poly_fp_powmod(base: &PolyFp, exp: &Integer, modulus: &PolyFp, p: u64) -> PolyFp {
    if modulus.is_empty() {
        return Vec::new();
    }
    let mut result: PolyFp = vec![1]; // constant 1
    let mut cur = poly_fp_rem(base, modulus, p);

    // Iterate over bits of exp from LSB to MSB
    let bit_len = exp.significant_bits();
    for bit_idx in 0..bit_len {
        if exp.get_bit(bit_idx) {
            result = poly_fp_mul(&result, &cur, p);
            result = poly_fp_rem(&result, modulus, p);
        }
        cur = poly_fp_mul(&cur, &cur, p);
        cur = poly_fp_rem(&cur, modulus, p);
    }
    result
}

/// Check if polynomial f has a factor of degree <= 2 over F_p.
///
/// Uses the fact that the product of all irreducible polynomials of degree
/// dividing k over F_p is x^{p^k} - x. So:
///   gcd(x^{p^1} - x, f) captures all degree-1 factors (roots)
///   gcd(x^{p^2} - x, f) captures all factors of degree 1 or 2
///
/// If f has no roots (degree-1 factors), we only need to check for degree-2 factors
/// by computing gcd(x^{p^2} - x, f) mod p and checking if its degree > 0.
pub fn has_factor_degree_le_2(f_coeffs: &[i64], p: u64) -> bool {
    let d = f_coeffs.len() - 1;
    if d <= 2 {
        return true; // trivially has a factor of degree <= deg(f)
    }

    // Convert f to F_p representation
    let f_fp: PolyFp = f_coeffs
        .iter()
        .map(|&c| ((c as i128).rem_euclid(p as i128)) as u64)
        .collect();

    // x = [0, 1] in ascending degree representation
    let x: PolyFp = vec![0, 1];

    // Compute x^{p^2} mod f over F_p
    // p^2 can be very large, so use Integer for the exponent
    let p_sq = Integer::from(p) * Integer::from(p);
    let x_p2 = poly_fp_powmod(&x, &p_sq, &f_fp, p);

    // Compute x^{p^2} - x mod f over F_p
    let mut diff = x_p2;
    // Subtract x: diff[1] -= 1
    if diff.len() < 2 {
        diff.resize(2, 0);
    }
    diff[1] = (diff[1] + p - 1) % p;
    poly_fp_normalize(&mut diff);

    // gcd(x^{p^2} - x, f) mod p
    let g = poly_fp_gcd(&diff, &f_fp, p);

    // If degree of gcd > 0, f has a factor of degree <= 2
    g.len() > 1
}

/// Quadratic character primes for GNFS: ensures algebraic product is a square in O_K.
/// Each (q, r) pair represents a prime q not in the factor base where f(r) ≡ 0 (mod q).
/// The Legendre symbol ((a - b*r) / q) must be +1 for all relations in a dependency
/// to guarantee the algebraic product has a square root in the number field.
pub struct QuadCharSet {
    pub primes: Vec<u64>,
    pub roots: Vec<u64>,
}

/// Select quadratic character primes: primes q not in the factor base where f has a root.
pub fn select_quad_char_primes(f_coeffs: &[i64], fb_primes: &[u64], count: usize) -> QuadCharSet {
    if count == 0 {
        return QuadCharSet {
            primes: Vec::new(),
            roots: Vec::new(),
        };
    }
    use std::collections::HashSet;
    let fb_set: HashSet<u64> = fb_primes.iter().cloned().collect();
    let mut result = QuadCharSet {
        primes: Vec::with_capacity(count),
        roots: Vec::with_capacity(count),
    };

    for &p in &sieve_primes(1_000_000) {
        if p < 3 || fb_set.contains(&p) {
            continue;
        }
        let roots = find_polynomial_roots_mod_p(f_coeffs, p);
        if !roots.is_empty() {
            // Push ALL roots for each QC prime. For degree-d polynomials,
            // each prime can have up to d roots; we need a QC column for
            // every (prime, root) pair to fully constrain the algebraic
            // product to be a square across all prime ideals above each prime.
            for &r in &roots {
                result.primes.push(p);
                result.roots.push(r);
            }
            // Count distinct primes, not total (prime, root) pairs
            let distinct_count = {
                let mut seen = std::collections::HashSet::new();
                result.primes.iter().for_each(|&p| { seen.insert(p); });
                seen.len()
            };
            if distinct_count >= count {
                break;
            }
        }
    }
    result
}

/// Compute the Legendre symbol (a/p) for a ≥ 0, p odd prime.
/// Returns 1 if a is a QR mod p, p-1 (i.e. -1 mod p) if QNR, 0 if p | a.
pub fn legendre_symbol(a: u64, p: u64) -> u64 {
    if a % p == 0 {
        return 0;
    }
    let a_int = Integer::from(a);
    let p_int = Integer::from(p);
    let exp = Integer::from((p - 1) / 2);
    let result = a_int.pow_mod(&exp, &p_int).unwrap();
    result.to_u64().unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rug::Integer;

    #[test]
    fn test_isqrt() {
        assert_eq!(isqrt(&Integer::from(0)), Integer::from(0));
        assert_eq!(isqrt(&Integer::from(1)), Integer::from(1));
        assert_eq!(isqrt(&Integer::from(4)), Integer::from(2));
        assert_eq!(isqrt(&Integer::from(10)), Integer::from(3));
        assert_eq!(isqrt(&Integer::from(100)), Integer::from(10));
        assert_eq!(isqrt(&Integer::from(8051)), Integer::from(89));
    }

    #[test]
    fn test_nth_root() {
        assert_eq!(nth_root(&Integer::from(8), 3), Integer::from(2));
        assert_eq!(nth_root(&Integer::from(27), 3), Integer::from(3));
        assert_eq!(nth_root(&Integer::from(1000), 3), Integer::from(10));
        assert_eq!(nth_root(&Integer::from(16), 4), Integer::from(2));
    }

    #[test]
    fn test_sieve_primes() {
        let primes = sieve_primes(30);
        assert_eq!(primes, vec![2, 3, 5, 7, 11, 13, 17, 19, 23, 29]);
    }

    #[test]
    fn test_sieve_primes_large() {
        let primes = sieve_primes(100);
        assert_eq!(primes.len(), 25);
        assert_eq!(primes[0], 2);
        assert_eq!(*primes.last().unwrap(), 97);
    }

    #[test]
    fn test_mod_inverse() {
        assert_eq!(mod_inverse(3, 10), Some(7));
        assert_eq!(mod_inverse(2, 10), None);
        assert_eq!(mod_inverse(3, 7), Some(5));
    }

    #[test]
    fn test_mod_inverse_u64() {
        assert_eq!(mod_inverse_u64(3, 10), Some(7));
        assert_eq!(mod_inverse_u64(2, 10), None);
        assert_eq!(mod_inverse_u64(3, 7), Some(5));
        // Verify consistency with rug-based version
        for m in [7u64, 13, 97, 1009, 65537] {
            for a in 1..m.min(50) {
                assert_eq!(
                    mod_inverse_u64(a, m),
                    mod_inverse(a, m),
                    "mismatch for a={}, m={}",
                    a,
                    m
                );
            }
        }
    }

    #[test]
    fn test_tonelli_shanks() {
        let r = tonelli_shanks(4, 7).unwrap();
        assert_eq!((r * r) % 7, 4);
        let r = tonelli_shanks(2, 7).unwrap();
        assert_eq!((r * r) % 7, 2);
        assert!(tonelli_shanks(3, 7).is_none());
    }

    #[test]
    fn test_polynomial_roots_mod_p() {
        let coeffs: Vec<i64> = vec![-1, 0, 1]; // x^2 - 1
        let roots = find_polynomial_roots_mod_p(&coeffs, 7);
        assert!(roots.contains(&1));
        assert!(roots.contains(&6));
    }

    #[test]
    fn test_eval_poly_i64() {
        let coeffs = vec![3i64, 2, 1]; // 3 + 2x + x^2
        assert_eq!(eval_poly_mod(&coeffs, 0, 100), 3);
        assert_eq!(eval_poly_mod(&coeffs, 1, 100), 6);
        assert_eq!(eval_poly_mod(&coeffs, 2, 100), 11);
    }
}
