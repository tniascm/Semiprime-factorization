/// Factor base construction for NFS: polynomial root finding, Tonelli-Shanks,
/// and per-prime trial divisor tables.
///
/// The factor base stores, for each prime p up to a bound, the roots of the
/// NFS polynomial mod p together with Montgomery-form trial divisors and
/// quantised log values for the line sieve.

use crate::arith::{sieve_primes, MontgomeryParams, TrialDivisor};

// ---------------------------------------------------------------------------
// Polynomial evaluation mod m
// ---------------------------------------------------------------------------

/// Evaluate polynomial `f(x) = sum_i f_coeffs[i] * x^i` modulo `m` using
/// Horner's method.
///
/// Coefficients are given in ascending order: `f_coeffs[0]` is the constant
/// term. Negative coefficients are handled by adding `m` before reduction.
///
/// # Panics
/// Panics if `m == 0`.
pub fn eval_poly_mod(f_coeffs: &[i64], x: u64, m: u64) -> u64 {
    assert!(m > 0, "eval_poly_mod: modulus must be > 0");
    if f_coeffs.is_empty() {
        return 0;
    }
    let m128 = m as u128;
    // Horner: process from highest-degree coefficient down to constant.
    let mut acc: u128 = 0;
    for &c in f_coeffs.iter().rev() {
        // acc = acc * x + c  (mod m)
        acc = (acc * x as u128) % m128;
        if c >= 0 {
            acc = (acc + c as u128) % m128;
        } else {
            // c is negative: compute |c| mod m, then subtract.
            let abs_c = ((-c) as u128) % m128;
            acc = (acc + m128 - abs_c) % m128;
        }
    }
    acc as u64
}

// ---------------------------------------------------------------------------
// Root finding mod p
// ---------------------------------------------------------------------------

/// Find all roots of `f(x) = 0 mod p` by exhaustive evaluation.
///
/// For the NFS factor base (bound typically 30K-65K) and polynomials of
/// degree 3-4, exhaustive search over `0..p` is fast enough for every
/// prime in the factor base.
pub fn find_roots_mod_p(f_coeffs: &[i64], p: u64) -> Vec<u64> {
    let mut roots = Vec::new();
    for x in 0..p {
        if eval_poly_mod(f_coeffs, x, p) == 0 {
            roots.push(x);
        }
    }
    roots
}

// ---------------------------------------------------------------------------
// Tonelli-Shanks square root mod p
// ---------------------------------------------------------------------------

/// Compute `sqrt(n) mod p` where `p` is an odd prime, returning one of the
/// two roots (or `None` if `n` is a quadratic non-residue mod p).
///
/// Uses Tonelli-Shanks with Montgomery arithmetic for all modular operations.
///
/// # Panics
/// Panics if `p < 3` or `p` is even.
pub fn tonelli_shanks(n: u64, p: u64) -> Option<u64> {
    assert!(p >= 3 && p & 1 == 1, "tonelli_shanks: p must be an odd prime >= 3");

    let n = n % p;
    if n == 0 {
        return Some(0);
    }

    let mont = MontgomeryParams::new(p);

    // Euler criterion: n^((p-1)/2) must be 1 for n to be a QR.
    let euler = mont.powmod(n, (p - 1) / 2);
    if euler != 1 {
        return None; // quadratic non-residue
    }

    // Factor p - 1 = q * 2^s with q odd.
    let mut q = p - 1;
    let mut s: u32 = 0;
    while q & 1 == 0 {
        q >>= 1;
        s += 1;
    }

    // Special case: p ≡ 3 (mod 4) => s == 1, root = n^((p+1)/4).
    if s == 1 {
        let r = mont.powmod(n, (p + 1) / 4);
        return Some(r);
    }

    // Find a quadratic non-residue z.
    let mut z = 2u64;
    while mont.powmod(z, (p - 1) / 2) != p - 1 {
        z += 1;
    }

    // Initialise.
    let mut m = s;
    let mut c = mont.powmod(z, q);       // z^q mod p
    let mut t = mont.powmod(n, q);       // n^q mod p
    let mut r = mont.powmod(n, (q + 1) / 2); // n^{(q+1)/2} mod p

    loop {
        if t == 1 {
            return Some(r);
        }

        // Find least i in 1..m such that t^{2^i} ≡ 1 (mod p).
        let mut i = 1u32;
        let mut tmp = mont.powmod(t, 2); // t^2 mod p
        while tmp != 1 {
            tmp = mont.powmod(tmp, 2);
            i += 1;
        }

        // Update: b = c^{2^{m-i-1}}
        let exp = 1u64 << (m - i - 1);
        let b = mont.powmod(c, exp);
        let b2 = mont.powmod(b, 2);

        // r = r * b, t = t * b^2, c = b^2, m = i
        r = (r as u128 * b as u128 % p as u128) as u64;
        t = (t as u128 * b2 as u128 % p as u128) as u64;
        c = b2;
        m = i;
    }
}

// ---------------------------------------------------------------------------
// FactorBase
// ---------------------------------------------------------------------------

/// Factor base for one side of NFS.
///
/// Contains all primes up to a given bound together with roots of the NFS
/// polynomial mod each prime, Montgomery-form trial divisors, and quantised
/// log values for sieve scoring.
#[derive(Debug, Clone)]
pub struct FactorBase {
    /// Primes in the factor base.
    pub primes: Vec<u64>,
    /// Per-prime: all roots of f(x) mod p.
    pub roots: Vec<Vec<u64>>,
    /// Montgomery-form trial divisors for fast divisibility checks.
    pub trial_divisors: Vec<TrialDivisor>,
    /// Quantised log2(p) values for sieve scoring.
    pub log_p: Vec<u8>,
    /// Scale factor used for log quantisation.
    pub scale: f64,
}

impl FactorBase {
    /// Build a factor base for polynomial `f` with primes up to `bound`.
    ///
    /// 1. Sieve primes up to `bound`.
    /// 2. For each prime `p`, find all roots of `f(x) mod p`.
    /// 3. Build a `TrialDivisor` for each prime.
    /// 4. Compute `log_p = floor(log2(p) * scale)` clamped to `u8`.
    pub fn new(f_coeffs: &[i64], bound: u64, scale: f64) -> Self {
        let primes = sieve_primes(bound);
        let mut roots = Vec::with_capacity(primes.len());
        let mut trial_divisors = Vec::with_capacity(primes.len());
        let mut log_p_vec = Vec::with_capacity(primes.len());

        for &p in &primes {
            roots.push(find_roots_mod_p(f_coeffs, p));
            trial_divisors.push(TrialDivisor::new(p, scale));

            let lp = ((p as f64).log2() * scale).floor();
            let lp_clamped = if lp < 0.0 {
                0u8
            } else if lp > 255.0 {
                255u8
            } else {
                lp as u8
            };
            log_p_vec.push(lp_clamped);
        }

        FactorBase {
            primes,
            roots,
            trial_divisors,
            log_p: log_p_vec,
            scale,
        }
    }

    /// Total number of (prime, root) pairs in the factor base.
    pub fn pair_count(&self) -> usize {
        self.roots.iter().map(|r| r.len()).sum()
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eval_poly_mod() {
        // f(x) = x^2 + 1, coeffs in ascending order: [1, 0, 1]
        let f = [1i64, 0, 1];
        // f(3) mod 7 = (9 + 1) mod 7 = 10 mod 7 = 3
        assert_eq!(eval_poly_mod(&f, 3, 7), 3);
        // f(0) mod 7 = 1
        assert_eq!(eval_poly_mod(&f, 0, 7), 1);
    }

    #[test]
    fn test_eval_poly_mod_negative_coeffs() {
        // f(x) = x^2 - 1, coeffs: [-1, 0, 1]
        let f = [-1i64, 0, 1];
        // f(1) mod 7 = 0
        assert_eq!(eval_poly_mod(&f, 1, 7), 0);
        // f(6) mod 7 = (36 - 1) mod 7 = 35 mod 7 = 0
        assert_eq!(eval_poly_mod(&f, 6, 7), 0);
        // f(2) mod 7 = (4 - 1) mod 7 = 3
        assert_eq!(eval_poly_mod(&f, 2, 7), 3);
    }

    #[test]
    fn test_find_roots_small() {
        // f(x) = x^2 - 1, roots mod 7 should be {1, 6}
        let f = [-1i64, 0, 1];
        let mut roots = find_roots_mod_p(&f, 7);
        roots.sort();
        assert_eq!(roots, vec![1, 6]);
    }

    #[test]
    fn test_find_roots_degree3() {
        // f(x) = x^3 + 2x + 1, coeffs: [1, 2, 0, 1]
        let f = [1i64, 2, 0, 1];
        let roots = find_roots_mod_p(&f, 5);
        // Verify each root
        for &r in &roots {
            assert_eq!(
                eval_poly_mod(&f, r, 5),
                0,
                "f({}) mod 5 should be 0",
                r
            );
        }
        // Manually verify: f(0)=1, f(1)=4, f(2)=13%5=3, f(3)=34%5=4, f(4)=73%5=3
        // No roots mod 5
        assert!(roots.is_empty(), "x^3 + 2x + 1 has no roots mod 5");
    }

    #[test]
    fn test_find_roots_degree3_has_roots() {
        // f(x) = x^3 + 2x + 1, check mod 7
        let f = [1i64, 2, 0, 1];
        let roots = find_roots_mod_p(&f, 7);
        for &r in &roots {
            assert_eq!(
                eval_poly_mod(&f, r, 7),
                0,
                "f({}) mod 7 should be 0",
                r
            );
        }
        // f(0)=1, f(1)=4, f(2)=13%7=6, f(3)=34%7=6, f(4)=73%7=3, f(5)=136%7=3, f(6)=229%7=5
        // No roots mod 7 either. Try mod 3:
        let roots3 = find_roots_mod_p(&f, 3);
        for &r in &roots3 {
            assert_eq!(eval_poly_mod(&f, r, 3), 0, "f({}) mod 3 should be 0", r);
        }
        // f(0)=1, f(1)=4%3=1, f(2)=13%3=1 => no roots mod 3
        // Try mod 2: f(0)=1, f(1)=4%2=0 => root at 1
        let roots2 = find_roots_mod_p(&f, 2);
        assert_eq!(roots2, vec![1]);
    }

    #[test]
    fn test_tonelli_shanks() {
        // sqrt(4) mod 7: should be 2 or 5
        let r = tonelli_shanks(4, 7).expect("4 is a QR mod 7");
        assert!(
            r == 2 || r == 5,
            "sqrt(4) mod 7 should be 2 or 5, got {}",
            r
        );
        // Verify: r^2 ≡ 4 (mod 7)
        assert_eq!((r * r) % 7, 4);

        // 3 mod 7 is a QNR
        assert!(
            tonelli_shanks(3, 7).is_none(),
            "3 is a QNR mod 7, should return None"
        );
    }

    #[test]
    fn test_tonelli_shanks_zero() {
        assert_eq!(tonelli_shanks(0, 7), Some(0));
        assert_eq!(tonelli_shanks(7, 7), Some(0));
    }

    #[test]
    fn test_tonelli_shanks_various_primes() {
        // Test across several primes with known QRs.
        // Each (n, p) pair: n must be a QR mod p.
        // QRs mod 5: {0,1,4}; mod 13: {0,1,3,4,9,10,12}; etc.
        let test_cases = [(4, 5), (4, 13), (9, 17), (16, 23), (25, 29)];
        for &(n, p) in &test_cases {
            let r = tonelli_shanks(n, p).unwrap_or_else(|| {
                panic!("{} should be a QR mod {}", n, p);
            });
            assert_eq!(
                (r as u128 * r as u128 % p as u128) as u64,
                n % p,
                "sqrt({}) mod {}: {} does not square back",
                n,
                p,
                r
            );
        }
    }

    #[test]
    fn test_tonelli_shanks_large_s() {
        // p = 97 has p-1 = 96 = 3 * 2^5, so s = 5 (exercises the main loop).
        // 4 is a QR mod 97 (since 2^48 mod 97 ≡ 1 via Euler criterion)
        let r = tonelli_shanks(4, 97).expect("4 is a QR mod 97");
        assert_eq!((r * r) % 97, 4, "sqrt(4) mod 97 failed: got {}", r);
    }

    #[test]
    fn test_factor_base_construction() {
        // f(x) = x^3 + 2x + 1, coeffs: [1, 2, 0, 1]
        let f = [1i64, 2, 0, 1];
        let fb = FactorBase::new(&f, 50, 1.0);

        // Primes start at 2
        assert_eq!(fb.primes[0], 2, "first prime should be 2");

        // All arrays have consistent lengths
        assert_eq!(fb.primes.len(), fb.roots.len());
        assert_eq!(fb.primes.len(), fb.trial_divisors.len());
        assert_eq!(fb.primes.len(), fb.log_p.len());

        // ALL roots are valid: eval_poly_mod(f, r, p) == 0
        for (i, &p) in fb.primes.iter().enumerate() {
            for &r in &fb.roots[i] {
                assert_eq!(
                    eval_poly_mod(&f, r, p),
                    0,
                    "root {} of f mod {} is invalid",
                    r,
                    p
                );
            }
        }
    }

    #[test]
    fn test_factor_base_large() {
        // f(x) = x^3 - x + 1, coeffs: [1, -1, 0, 1]
        let f = [1i64, -1, 0, 1];
        let fb = FactorBase::new(&f, 1000, 1.0);

        assert!(fb.pair_count() > 0, "should have at least some roots");

        // All roots valid
        for (i, &p) in fb.primes.iter().enumerate() {
            for &r in &fb.roots[i] {
                assert_eq!(
                    eval_poly_mod(&f, r, p),
                    0,
                    "root {} of f mod {} is invalid",
                    r,
                    p
                );
            }
        }
    }

    #[test]
    fn test_factor_base_pair_count() {
        // f(x) = x^2 - 1 should have 2 roots mod most odd primes (1 and p-1)
        let f = [-1i64, 0, 1];
        let fb = FactorBase::new(&f, 50, 1.0);
        // At least as many pairs as primes with roots
        assert!(fb.pair_count() > 0);
        // Each prime p > 2 should have exactly 2 roots (1 and p-1) for x^2 - 1
        for (i, &p) in fb.primes.iter().enumerate() {
            if p > 2 {
                assert_eq!(
                    fb.roots[i].len(),
                    2,
                    "x^2 - 1 should have 2 roots mod {} but got {:?}",
                    p,
                    fb.roots[i]
                );
            }
        }
    }

    #[test]
    fn test_factor_base_log_p_values() {
        let f = [1i64, 0, 1]; // x^2 + 1
        let scale = 4.0;
        let fb = FactorBase::new(&f, 20, scale);

        // Verify log_p values are reasonable
        for (i, &p) in fb.primes.iter().enumerate() {
            let expected = ((p as f64).log2() * scale).floor() as u8;
            assert_eq!(
                fb.log_p[i], expected,
                "log_p for prime {} should be {}",
                p, expected
            );
        }
    }
}
