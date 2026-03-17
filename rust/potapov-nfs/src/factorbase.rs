/// Factor base construction for NFS: polynomial root finding, Tonelli-Shanks,
/// and per-prime trial divisor tables.
///
/// The factor base stores, for each prime p up to a bound, the roots of the
/// NFS polynomial mod p together with Montgomery-form trial divisors and
/// quantised log values for the line sieve.
use crate::arith::{mod_inverse, sieve_primes, MontgomeryParams, TrialDivisor};

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

/// Find all roots of `f(x) = 0 mod p`.
///
/// Fast paths:
/// - degree 1: direct modular inverse
/// - degree 2: quadratic formula + Tonelli-Shanks
///
/// For degree >= 3 we evaluate over `x=0..p-1` via forward differences
/// (additions only), which is substantially faster than repeated Horner
/// evaluation for each `x`.
pub fn find_roots_mod_p(f_coeffs: &[i64], p: u64) -> Vec<u64> {
    if p == 0 || f_coeffs.is_empty() {
        return Vec::new();
    }

    let degree = match leading_degree(f_coeffs) {
        Some(d) => d,
        None => return (0..p).collect(), // zero polynomial
    };

    if degree == 0 {
        return if coeff_mod_p(f_coeffs[0], p) == 0 {
            (0..p).collect()
        } else {
            Vec::new()
        };
    }

    if degree == 1 {
        let c0 = coeff_mod_p(f_coeffs[0], p);
        let c1 = coeff_mod_p(f_coeffs[1], p);
        if c1 == 0 {
            return if c0 == 0 {
                (0..p).collect()
            } else {
                Vec::new()
            };
        }
        let inv = match mod_inverse(c1, p) {
            Some(v) => v,
            None => return Vec::new(),
        };
        let root = (((p - c0) as u128 * inv as u128) % p as u128) as u64;
        return vec![root];
    }

    if degree == 2 {
        let c0 = coeff_mod_p(f_coeffs[0], p);
        let c1 = coeff_mod_p(f_coeffs[1], p);
        let c2 = coeff_mod_p(f_coeffs[2], p);
        if c2 == 0 {
            // Degenerated quadratic -> linear.
            if c1 == 0 {
                return if c0 == 0 {
                    (0..p).collect()
                } else {
                    Vec::new()
                };
            }
            let inv = match mod_inverse(c1, p) {
                Some(v) => v,
                None => return Vec::new(),
            };
            let root = (((p - c0) as u128 * inv as u128) % p as u128) as u64;
            return vec![root];
        }

        // p is prime in our usage, but handle p=2 robustly by direct check.
        if p == 2 {
            let mut roots = Vec::new();
            for x in 0..2 {
                if eval_poly_mod(f_coeffs, x, p) == 0 {
                    roots.push(x);
                }
            }
            return roots;
        }

        // x = (-b ± sqrt(b^2 - 4ac)) / (2a) mod p
        let p128 = p as u128;
        let b2 = (c1 as u128 * c1 as u128) % p128;
        let four_ac = (4u128 * c2 as u128 % p128) * c0 as u128 % p128;
        let disc = ((b2 + p128) - four_ac) % p128;
        let sqrt_disc = match tonelli_shanks(disc as u64, p) {
            Some(v) => v,
            None => return Vec::new(),
        };
        let denom = ((2u128 * c2 as u128) % p128) as u64;
        let denom_inv = match mod_inverse(denom, p) {
            Some(v) => v,
            None => return Vec::new(),
        };

        let minus_b = (p - c1) % p;
        let x1 = ((minus_b as u128 + sqrt_disc as u128) * denom_inv as u128 % p128) as u64;
        let x2 = ((minus_b as u128 + p128 - sqrt_disc as u128) * denom_inv as u128 % p128) as u64;
        if x1 == x2 {
            return vec![x1];
        }
        let (a, b) = if x1 < x2 { (x1, x2) } else { (x2, x1) };
        return vec![a, b];
    }

    // Degree >= 3: use Cantor-Zassenhaus to find roots in O(d^2 log p)
    // instead of brute-force O(p).
    //
    // 1. Reduce f mod p to get a monic polynomial over F_p
    // 2. Compute gcd(x^p - x, f(x)) mod p to get product of all linear factors
    // 3. Split the result using random elements
    find_roots_cantor_zassenhaus(f_coeffs, p)
}

// ---------------------------------------------------------------------------
// Polynomial arithmetic mod p (for Cantor-Zassenhaus)
// ---------------------------------------------------------------------------

/// Polynomial over F_p, stored as Vec<u64> in ascending degree order.
/// poly[i] = coefficient of x^i. The zero polynomial is represented as [].
type Poly = Vec<u64>;

/// Reduce polynomial by stripping trailing zeros.
fn poly_normalize(a: &mut Poly) {
    while a.last() == Some(&0) {
        a.pop();
    }
}

/// Polynomial multiplication mod p.
fn poly_mul_mod(a: &Poly, b: &Poly, p: u64) -> Poly {
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
    poly_normalize(&mut c);
    c
}

/// Polynomial remainder: a mod b, over F_p. Returns a mod b.
fn poly_rem(a: &Poly, b: &Poly, p: u64) -> Poly {
    if b.is_empty() {
        return a.clone(); // division by zero polynomial
    }
    let mut r = a.clone();
    let db = b.len() - 1;
    let lead_b_inv = mod_inverse(*b.last().unwrap(), p).unwrap();
    while r.len() > db {
        let coeff = ((r.last().copied().unwrap_or(0) as u128 * lead_b_inv as u128) % p as u128) as u64;
        let shift = r.len() - 1 - db;
        for (i, &bi) in b.iter().enumerate() {
            let sub = (coeff as u128 * bi as u128) % p as u128;
            r[shift + i] = ((r[shift + i] as u128 + p as u128 - sub) % p as u128) as u64;
        }
        poly_normalize(&mut r);
    }
    r
}

/// Polynomial GCD over F_p.
fn poly_gcd(a: &Poly, b: &Poly, p: u64) -> Poly {
    let mut a = a.clone();
    let mut b = b.clone();
    while !b.is_empty() {
        let r = poly_rem(&a, &b, p);
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

/// Compute x^n mod f(x) over F_p using repeated squaring.
fn poly_powmod(base: &Poly, mut exp: u64, modulus: &Poly, p: u64) -> Poly {
    if modulus.is_empty() {
        return Vec::new();
    }
    let mut result: Poly = vec![1]; // constant 1
    let mut cur = poly_rem(base, modulus, p);
    while exp > 0 {
        if exp & 1 == 1 {
            result = poly_mul_mod(&result, &cur, p);
            result = poly_rem(&result, modulus, p);
        }
        cur = poly_mul_mod(&cur, &cur, p);
        cur = poly_rem(&cur, modulus, p);
        exp >>= 1;
    }
    result
}

/// Find all roots of f(x) = 0 mod p using Cantor-Zassenhaus algorithm.
///
/// Complexity: O(d^2 log p) expected, where d = deg(f).
/// This replaces the O(p) brute-force evaluation for degree >= 3.
fn find_roots_cantor_zassenhaus(f_coeffs: &[i64], p: u64) -> Vec<u64> {
    // Convert to Poly over F_p
    let mut f: Poly = f_coeffs.iter().map(|&c| coeff_mod_p(c, p)).collect();
    poly_normalize(&mut f);
    if f.is_empty() {
        return (0..p).collect();
    }

    // For very small p (p <= degree), brute force is faster
    let degree = f.len() - 1;
    if p <= (degree as u64) * 4 {
        let mut roots = Vec::new();
        for x in 0..p {
            if eval_poly_mod(f_coeffs, x, p) == 0 {
                roots.push(x);
            }
        }
        return roots;
    }

    // Step 1: Compute g = gcd(x^p - x, f) to get the product of all linear factors.
    // x^p - x mod f is computed via modular exponentiation.
    let x_poly: Poly = vec![0, 1]; // the polynomial "x"
    let xp = poly_powmod(&x_poly, p, &f, p);
    // xp - x mod p
    let mut xp_minus_x = xp;
    if xp_minus_x.len() < 2 {
        xp_minus_x.resize(2, 0);
    }
    xp_minus_x[1] = (xp_minus_x[1] + p - 1) % p;
    poly_normalize(&mut xp_minus_x);

    let g = poly_gcd(&f, &xp_minus_x, p);
    if g.is_empty() || g.len() == 1 {
        // g is constant => no roots
        return Vec::new();
    }

    // g is the product of all distinct linear factors of f mod p.
    // Now extract all roots by splitting g.
    let mut roots = Vec::new();
    split_roots(&g, p, &mut roots, 0);
    roots.sort_unstable();
    roots
}

/// Recursively split a square-free polynomial (product of linear factors) into roots.
fn split_roots(g: &Poly, p: u64, roots: &mut Vec<u64>, seed: u64) {
    let deg = g.len() - 1;
    if deg == 0 {
        return;
    }
    if deg == 1 {
        // g(x) = x - r (monic), so root = p - g[0]
        let r = (p - g[0]) % p;
        roots.push(r);
        return;
    }

    // For degree 2, use quadratic formula directly
    if deg == 2 {
        // g = x^2 + bx + c (monic after normalization in poly_gcd)
        let b = g[1];
        let c = g[0];
        let p128 = p as u128;
        let disc = ((b as u128 * b as u128 % p128) + p128 - (4u128 * c as u128 % p128)) % p128;
        if let Some(sq) = crate::factorbase::tonelli_shanks(disc as u64, p) {
            let inv2 = mod_inverse(2, p).unwrap();
            let neg_b = (p - b) % p;
            let r1 = ((neg_b as u128 + sq as u128) % p128 * inv2 as u128 % p128) as u64;
            let r2 = ((neg_b as u128 + p128 - sq as u128) % p128 * inv2 as u128 % p128) as u64;
            roots.push(r1);
            if r1 != r2 {
                roots.push(r2);
            }
        }
        return;
    }

    // Cantor-Zassenhaus splitting: pick random a, compute gcd(g, (x+a)^((p-1)/2) - 1)
    let half_p = (p - 1) / 2;
    let mut rng = seed;
    for attempt in 0..100 {
        // Simple LCG for deterministic randomness
        rng = rng
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let a = (rng >> 17) % p;

        // Compute (x + a)^((p-1)/2) mod g
        let xa: Poly = vec![a, 1]; // x + a
        let powered = poly_powmod(&xa, half_p, g, p);
        // powered - 1
        let mut pm1 = powered;
        if pm1.is_empty() {
            pm1.push(p - 1);
        } else {
            pm1[0] = (pm1[0] + p - 1) % p;
        }
        poly_normalize(&mut pm1);

        let h = poly_gcd(g, &pm1, p);
        let h_deg = if h.is_empty() { 0 } else { h.len() - 1 };

        if h_deg > 0 && h_deg < deg {
            // Successful split
            split_roots(&h, p, roots, rng.wrapping_add(attempt));
            let q = poly_exact_div(g, &h, p);
            split_roots(&q, p, roots, rng.wrapping_add(attempt + 1000));
            return;
        }
    }

    // Fallback: brute force (should not happen for reasonable inputs)
    for x in 0..p {
        let mut val = 0u128;
        let mut x_pow = 1u128;
        for &c in g.iter() {
            val = (val + c as u128 * x_pow) % p as u128;
            x_pow = x_pow * x as u128 % p as u128;
        }
        if val == 0 {
            roots.push(x);
        }
    }
}

/// Exact polynomial division: a / b over F_p, assuming b divides a.
fn poly_exact_div(a: &Poly, b: &Poly, p: u64) -> Poly {
    if b.is_empty() || a.is_empty() {
        return Vec::new();
    }
    let da = a.len() - 1;
    let db = b.len() - 1;
    if da < db {
        return Vec::new();
    }
    let mut r = a.clone();
    let lead_b_inv = mod_inverse(*b.last().unwrap(), p).unwrap();
    let mut q = vec![0u64; da - db + 1];
    for i in (0..=(da - db)).rev() {
        let coeff = (r[i + db] as u128 * lead_b_inv as u128 % p as u128) as u64;
        q[i] = coeff;
        for (j, &bj) in b.iter().enumerate() {
            let sub = (coeff as u128 * bj as u128) % p as u128;
            r[i + j] = ((r[i + j] as u128 + p as u128 - sub) % p as u128) as u64;
        }
    }
    poly_normalize(&mut q);
    q
}

#[inline]
fn coeff_mod_p(c: i64, p: u64) -> u64 {
    (c as i128).rem_euclid(p as i128) as u64
}

fn leading_degree(f_coeffs: &[i64]) -> Option<usize> {
    for i in (0..f_coeffs.len()).rev() {
        if f_coeffs[i] != 0 {
            return Some(i);
        }
    }
    None
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
    assert!(
        p >= 3 && p & 1 == 1,
        "tonelli_shanks: p must be an odd prime >= 3"
    );

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
    let mut c = mont.powmod(z, q); // z^q mod p
    let mut t = mont.powmod(n, q); // n^q mod p
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

        // Parallelize root-finding — the expensive per-prime operation.
        use rayon::prelude::*;
        let per_prime: Vec<_> = primes
            .par_iter()
            .map(|&p| {
                let r = find_roots_mod_p(f_coeffs, p);
                let td = TrialDivisor::new(p, scale);
                let lp = ((p as f64).log2() * scale).floor();
                let lp_clamped = if lp < 0.0 { 0u8 } else if lp > 255.0 { 255u8 } else { lp as u8 };
                (r, td, lp_clamped)
            })
            .collect();

        let mut roots = Vec::with_capacity(primes.len());
        let mut trial_divisors = Vec::with_capacity(primes.len());
        let mut log_p_vec = Vec::with_capacity(primes.len());
        for (r, td, lp) in per_prime {
            roots.push(r);
            trial_divisors.push(td);
            log_p_vec.push(lp);
        }

        FactorBase {
            primes,
            roots,
            trial_divisors,
            log_p: log_p_vec,
            scale,
        }
    }

    /// Build a factor base containing ONLY primes where `f` has at least one root mod p.
    ///
    /// Primes without roots (inert or partially-split with no degree-1 ideals)
    /// are excluded. This matches the gnfs matrix's algebraic column layout,
    /// which only tracks degree-1 prime ideals.
    pub fn new_roots_only(f_coeffs: &[i64], bound: u64, scale: f64) -> Self {
        let full_primes = sieve_primes(bound);

        // Parallelize root-finding, then filter primes with roots.
        use rayon::prelude::*;
        let per_prime: Vec<_> = full_primes
            .par_iter()
            .map(|&p| {
                let r = find_roots_mod_p(f_coeffs, p);
                (p, r)
            })
            .collect();

        let mut primes = Vec::new();
        let mut roots = Vec::new();
        let mut trial_divisors = Vec::new();
        let mut log_p = Vec::new();

        for (p, r) in per_prime {
            if !r.is_empty() {
                let lp = ((p as f64).log2() * scale).floor();
                let lp_clamped = if lp < 0.0 { 0u8 } else if lp > 255.0 { 255u8 } else { lp as u8 };
                primes.push(p);
                roots.push(r);
                trial_divisors.push(TrialDivisor::new(p, scale));
                log_p.push(lp_clamped);
            }
        }

        FactorBase {
            primes,
            roots,
            trial_divisors,
            log_p,
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
            assert_eq!(eval_poly_mod(&f, r, 5), 0, "f({}) mod 5 should be 0", r);
        }
        // Manually verify: f(0)=1, f(1)=4, f(2)=13%5=3, f(3)=34%5=4, f(4)=73%5=3
        // No roots mod 5
        assert!(roots.is_empty(), "x^3 + 2x + 1 has no roots mod 5");
    }

    #[test]
    fn test_find_roots_linear() {
        // f(x) = x - 3 over mod 11 -> root at x=3
        let f = [-3i64, 1];
        let roots = find_roots_mod_p(&f, 11);
        assert_eq!(roots, vec![3]);
    }

    #[test]
    fn test_find_roots_quadratic() {
        // f(x) = x^2 - 1 over mod 11 -> roots at 1 and 10
        let f = [-1i64, 0, 1];
        let roots = find_roots_mod_p(&f, 11);
        assert_eq!(roots, vec![1, 10]);
    }

    #[test]
    fn test_find_roots_degree3_has_roots() {
        // f(x) = x^3 + 2x + 1, check mod 7
        let f = [1i64, 2, 0, 1];
        let roots = find_roots_mod_p(&f, 7);
        for &r in &roots {
            assert_eq!(eval_poly_mod(&f, r, 7), 0, "f({}) mod 7 should be 0", r);
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
