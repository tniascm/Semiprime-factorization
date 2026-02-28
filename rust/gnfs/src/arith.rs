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
