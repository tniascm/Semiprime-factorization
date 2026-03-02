/// Montgomery u64 arithmetic, trial division, Miller-Rabin primality, and supporting utilities.
///
/// All operations use u64 exclusively (with u128 intermediates where needed for overflow).
/// No BigUint dependency.

// ---------------------------------------------------------------------------
// Montgomery reduction (REDC)
// ---------------------------------------------------------------------------

/// REDC: given t < n * 2^64, returns t * 2^{-64} mod n.
#[inline]
fn monty_reduce(t: u128, n: u64, n_inv: u64) -> u64 {
    let m = (t as u64).wrapping_mul(n_inv) as u128;
    let result = ((t + m * n as u128) >> 64) as u64;
    if result >= n {
        result - n
    } else {
        result
    }
}

// ---------------------------------------------------------------------------
// MontgomeryParams
// ---------------------------------------------------------------------------

/// Precomputed Montgomery parameters for an odd modulus `n`.
#[derive(Clone, Debug)]
pub struct MontgomeryParams {
    pub n: u64,
    /// -n^{-1} mod 2^{64}
    pub n_inv: u64,
    /// 2^{64} mod n
    pub r_mod_n: u64,
    /// (2^{64})^2 mod n
    pub r2_mod_n: u64,
}

impl MontgomeryParams {
    /// Compute Montgomery parameters for odd modulus `n`.
    ///
    /// # Panics
    /// Panics if `n` is even or less than 2.
    pub fn new(n: u64) -> Self {
        assert!(n >= 2 && n & 1 == 1, "MontgomeryParams: n must be odd and >= 2");

        // Compute n^{-1} mod 2^{64} via Newton's method.
        // Start with x = n (since n*n ≡ n^2 mod 2^k and for any odd n,
        // n itself is its own inverse mod 2 — actually we use the standard
        // Newton lift: x_{i+1} = x_i * (2 - n * x_i) which doubles the
        // number of correct bits each step.)
        //
        // Starting with x = 1 works because n * 1 ≡ 1 (mod 2) when n is odd.
        // After 6 iterations we have 2^6 = 64 correct bits.
        let mut x: u64 = 1; // n * 1 ≡ 1 mod 2 for odd n
        for _ in 0..6 {
            x = x.wrapping_mul(2u64.wrapping_sub(n.wrapping_mul(x)));
        }
        // x is now n^{-1} mod 2^{64}; we want n_inv = -n^{-1} mod 2^{64}
        let n_inv = x.wrapping_neg();

        // R mod n = 2^{64} mod n.
        // Since 2^{64} doesn't fit in u64, compute via u128.
        let r_mod_n = ((1u128 << 64) % n as u128) as u64;

        // R^2 mod n = (2^{64} mod n)^2 mod n
        let r2_mod_n = ((r_mod_n as u128 * r_mod_n as u128) % n as u128) as u64;

        MontgomeryParams { n, n_inv, r_mod_n, r2_mod_n }
    }

    /// Convert `a` (in [0, n)) to Montgomery form: a * R mod n.
    #[inline]
    pub fn to_mont(&self, a: u64) -> u64 {
        monty_reduce(a as u128 * self.r2_mod_n as u128, self.n, self.n_inv)
    }

    /// Convert from Montgomery form back to normal: aR * R^{-1} mod n.
    #[inline]
    pub fn from_mont(&self, ar: u64) -> u64 {
        monty_reduce(ar as u128, self.n, self.n_inv)
    }

    /// Montgomery multiplication: (aR)(bR) R^{-1} = abR mod n.
    #[inline]
    pub fn mul(&self, ar: u64, br: u64) -> u64 {
        monty_reduce(ar as u128 * br as u128, self.n, self.n_inv)
    }

    /// Montgomery squaring.
    #[inline]
    pub fn sqr(&self, ar: u64) -> u64 {
        self.mul(ar, ar)
    }

    /// Compute `ar^e mod n` where `ar` is already in Montgomery form.
    ///
    /// Returns the result in Montgomery form. This avoids repeated
    /// `to_mont`/`from_mont` conversions when chaining exponentiations
    /// (e.g. in the P-1 factoring stage-1 loop).
    pub fn powmod_mont(&self, ar: u64, e: u64) -> u64 {
        if e == 0 {
            return self.r_mod_n; // 1 in Montgomery form
        }
        let top = 63 - e.leading_zeros() as u32;
        let mut result = self.r_mod_n; // 1 in Montgomery form
        for i in (0..=top).rev() {
            result = self.sqr(result);
            if (e >> i) & 1 == 1 {
                result = self.mul(result, ar);
            }
        }
        result
    }

    /// Compute `a^e mod n` using binary exponentiation with Montgomery form.
    pub fn powmod(&self, a: u64, e: u64) -> u64 {
        if e == 0 {
            return 1 % self.n;
        }
        let base = self.to_mont(a % self.n);
        let one_mont = self.r_mod_n; // 1 * R mod n

        // Find the highest set bit position
        let top = 63 - e.leading_zeros() as u32; // bit index of MSB

        let mut result = one_mont;
        // Process from highest bit down to 0
        for i in (0..=top).rev() {
            result = self.sqr(result);
            if (e >> i) & 1 == 1 {
                result = self.mul(result, base);
            }
        }
        self.from_mont(result)
    }
}

// ---------------------------------------------------------------------------
// TrialDivisor — fast divisibility without division
// ---------------------------------------------------------------------------

/// Fast trial-divisibility test for a fixed odd prime `p`.
///
/// Uses the "inverse-multiply" trick: `n` is divisible by `p` iff
/// `n * p^{-1} mod 2^{64} <= floor(2^{64}-1 / p)`.
#[derive(Clone, Debug)]
pub struct TrialDivisor {
    pub p: u64,
    /// p^{-1} mod 2^{64} (only meaningful for odd p)
    pub p_inv: u64,
    /// floor((2^{64} - 1) / p)
    pub p_lim: u64,
    /// quantized log_2(p) * scale, clamped to u8
    pub log_p: u8,
}

impl TrialDivisor {
    /// Build a `TrialDivisor` for prime `p`.
    ///
    /// `scale` controls the log quantisation (e.g. 256.0 / 64.0 for a
    /// sieve-log byte).
    ///
    /// For `p == 2` the inverse trick does not apply (2 has no inverse
    /// mod 2^{64}); we store sentinel values and handle it in `divides`.
    pub fn new(p: u64, scale: f64) -> Self {
        let log_p = ((p as f64).log2() * scale) as u8;

        if p == 2 {
            return TrialDivisor {
                p: 2,
                p_inv: 0, // sentinel
                p_lim: 0, // sentinel
                log_p,
            };
        }

        // Newton's method for p^{-1} mod 2^{64} (same iteration as Montgomery)
        let mut x: u64 = 1;
        for _ in 0..6 {
            x = x.wrapping_mul(2u64.wrapping_sub(p.wrapping_mul(x)));
        }
        let p_inv = x;
        let p_lim = u64::MAX / p;

        TrialDivisor { p, p_inv, p_lim, log_p }
    }

    /// Returns `true` if `p | n`.
    #[inline]
    pub fn divides(&self, n: u64) -> bool {
        if self.p == 2 {
            return n & 1 == 0;
        }
        n.wrapping_mul(self.p_inv) <= self.p_lim
    }
}

// ---------------------------------------------------------------------------
// Miller-Rabin (base 2)
// ---------------------------------------------------------------------------

/// Deterministic Miller-Rabin test with witness 2.
///
/// Returns `true` if `n` is a probable prime (to base 2), `false` if
/// definitely composite.  For n < 2^{64}, base-2 MR alone has very few
/// false positives but is not fully deterministic; for NFS sieve usage
/// this suffices.
pub fn is_probable_prime(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    if n == 2 || n == 3 {
        return true;
    }
    if n & 1 == 0 {
        return false;
    }

    // Write n-1 = d * 2^r with d odd
    let mut d = n - 1;
    let mut r = 0u32;
    while d & 1 == 0 {
        d >>= 1;
        r += 1;
    }

    let mont = MontgomeryParams::new(n);
    let mut a = mont.powmod(2, d); // 2^d mod n

    if a == 1 || a == n - 1 {
        return true;
    }

    for _ in 0..r - 1 {
        a = mont.powmod(a, 2); // a^2 mod n
        if a == n - 1 {
            return true;
        }
    }

    false
}

// ---------------------------------------------------------------------------
// Sieve of Eratosthenes
// ---------------------------------------------------------------------------

/// Classic sieve returning all primes <= `bound`.
pub fn sieve_primes(bound: u64) -> Vec<u64> {
    if bound < 2 {
        return Vec::new();
    }
    let n = bound as usize;
    let mut is_prime = vec![true; n + 1];
    is_prime[0] = false;
    is_prime[1] = false;

    let mut i = 2;
    while i * i <= n {
        if is_prime[i] {
            let mut j = i * i;
            while j <= n {
                is_prime[j] = false;
                j += i;
            }
        }
        i += 1;
    }

    is_prime
        .iter()
        .enumerate()
        .filter_map(|(idx, &flag)| if flag { Some(idx as u64) } else { None })
        .collect()
}

// ---------------------------------------------------------------------------
// Extended GCD and modular inverse
// ---------------------------------------------------------------------------

/// Extended GCD: returns `(gcd, x)` where `a * x ≡ gcd (mod m)`.
///
/// Uses the signed-integer extended Euclidean algorithm on i128 to avoid
/// overflow.
pub fn extended_gcd(a: u64, m: u64) -> (u64, i64) {
    let (mut old_r, mut r) = (a as i128, m as i128);
    let (mut old_s, mut s) = (1i128, 0i128);

    while r != 0 {
        let q = old_r / r;
        let tmp = r;
        r = old_r - q * r;
        old_r = tmp;

        let tmp = s;
        s = old_s - q * s;
        old_s = tmp;
    }

    // old_r = gcd, old_s = x (may be negative)
    let gcd = old_r as u64;
    // Reduce x modulo m to get a value in [0, m)
    let x = ((old_s % m as i128) + m as i128) % m as i128;
    (gcd, x as i64)
}

/// Modular inverse: returns `a^{-1} mod m` if `gcd(a, m) == 1`, else `None`.
pub fn mod_inverse(a: u64, m: u64) -> Option<u64> {
    let (g, x) = extended_gcd(a, m);
    if g == 1 {
        Some(x as u64)
    } else {
        None
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_montgomery_roundtrip() {
        let mont = MontgomeryParams::new(97);
        for a in 0..97u64 {
            let ar = mont.to_mont(a);
            let back = mont.from_mont(ar);
            assert_eq!(back, a, "roundtrip failed for a={a}");
        }
    }

    #[test]
    fn test_montgomery_mul() {
        let mont = MontgomeryParams::new(97);
        for a in 1..97u64 {
            for b in 1..97u64 {
                let ar = mont.to_mont(a);
                let br = mont.to_mont(b);
                let cr = mont.mul(ar, br);
                let c = mont.from_mont(cr);
                assert_eq!(c, (a * b) % 97, "mul failed for a={a}, b={b}");
            }
        }
    }

    #[test]
    fn test_montgomery_powmod() {
        let mont = MontgomeryParams::new(97);
        // Fermat's little theorem: a^{96} ≡ 1 mod 97 for all a coprime to 97
        for a in 1..97u64 {
            assert_eq!(mont.powmod(a, 96), 1, "Fermat failed for a={a}");
        }
        // Specific case: 3^5 = 243 = 2*97 + 49
        assert_eq!(mont.powmod(3, 5), 49);
    }

    #[test]
    fn test_trial_divisor() {
        let td = TrialDivisor::new(7, 1.0);
        // Multiples of 7
        assert!(td.divides(0), "7 | 0");
        assert!(td.divides(7), "7 | 7");
        assert!(td.divides(14), "7 | 14");
        assert!(td.divides(49), "7 | 49");
        assert!(td.divides(7 * 1_000_000_007), "7 | 7*1000000007");
        // Non-multiples
        assert!(!td.divides(8), "7 nmid 8");
        assert!(!td.divides(15), "7 nmid 15");
        assert!(!td.divides(7 * 1_000_000_007 + 1), "7 nmid 7*1000000007+1");
    }

    #[test]
    fn test_is_probable_prime() {
        assert!(is_probable_prime(2));
        assert!(is_probable_prime(3));
        assert!(!is_probable_prime(4));
        assert!(is_probable_prime(131071)); // Mersenne prime 2^17 - 1
        assert!(is_probable_prime(1_000_000_007));
    }

    #[test]
    fn test_mod_inverse() {
        // 3 * 65 = 195 = 2*97 + 1 ≡ 1 mod 97
        assert_eq!(mod_inverse(3, 97), Some(65));
        // gcd(2, 4) = 2 > 1 → no inverse
        assert_eq!(mod_inverse(2, 4), None);
    }

    #[test]
    fn test_sieve_primes() {
        assert_eq!(
            sieve_primes(30),
            vec![2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
        );
    }

    #[test]
    fn test_montgomery_large_modulus() {
        // n = 2^{62} - 57  (a known prime)
        let n: u64 = (1u64 << 62) - 57;
        let mont = MontgomeryParams::new(n);
        // (n-1)^2 ≡ 1 mod n
        let result = mont.powmod(n - 1, 2);
        assert_eq!(result, 1, "(n-1)^2 mod n should be 1 for n = 2^62 - 57");
    }
}
