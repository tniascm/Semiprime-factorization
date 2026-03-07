//! Elliptic Curve Method (ECM) with u64 Montgomery curve arithmetic.
//!
//! Uses Montgomery curves `By^2 = x^3 + Ax^2 + x` in projective
//! coordinates `(X : Z)`, which allows differential addition and doubling
//! without computing `y`.  The Suyama parameterization gives curves with
//! a torsion point of order 12, improving the probability of finding
//! smooth group orders.
//!
//! For cofactors < 2^64 this is much faster than BigUint ECM.

use crate::arith::{mod_inverse, sieve_primes, MontgomeryParams};

/// A point on a Montgomery curve in projective coordinates `(X : Z)`.
#[derive(Clone, Copy, Debug)]
struct Point {
    x: u64,
    z: u64,
}

/// Precomputed curve parameter `a24 = (A + 2) / 4` in Montgomery form.
#[derive(Clone, Copy, Debug)]
struct CurveParams {
    a24: u64, // in Montgomery form
}

/// Run one ECM curve with given B1, B2 bounds and parameter `sigma`.
///
/// Uses Suyama parameterization to build a Montgomery curve with a
/// 12-torsion starting point, then performs a Montgomery-ladder scalar
/// multiplication for Stage 1 and a simple sequential Stage 2.
///
/// Returns `Some(factor)` if a non-trivial factor of `n` is found.
pub fn ecm_one_curve(n: u64, b1: u64, b2: u64, sigma: u64) -> Option<u64> {
    let primes = sieve_primes(b2);
    ecm_one_curve_with_primes(n, b1, b2, sigma, &primes)
}

/// ECM with pre-computed prime list (avoids redundant sieve_primes calls).
pub fn ecm_one_curve_with_primes(n: u64, b1: u64, b2: u64, sigma: u64, primes: &[u64]) -> Option<u64> {
    if n <= 1 || n % 2 == 0 {
        return None;
    }

    let mp = MontgomeryParams::new(n);

    // --- Suyama parameterization ---
    // u = sigma^2 - 5 mod n
    // v = 4 * sigma mod n
    let sigma_m = sigma % n;
    let u = submod(mulmod(sigma_m, sigma_m, n), 5, n);
    let v = mulmod(4, sigma_m, n);

    if u == 0 || v == 0 {
        return None;
    }

    // Starting point Q = (u^3 : v^3) on curve with
    // A = (v - u)^3 * (3u + v) / (4 * u^3 * v) - 2
    let u3 = mulmod(mulmod(u, u, n), u, n);
    let v3 = mulmod(mulmod(v, v, n), v, n);

    // Compute A+2 = (v - u)^3 * (3u + v) / (4 * u^3 * v)
    // Then a24 = (A + 2) / 4 = (v - u)^3 * (3u + v) / (16 * u^3 * v)
    let v_minus_u = submod(v, u, n);
    let vmu3 = mulmod(mulmod(v_minus_u, v_minus_u, n), v_minus_u, n);
    let three_u_plus_v = addmod(mulmod(3, u, n), v, n);
    let numerator = mulmod(vmu3, three_u_plus_v, n);

    let denom = mulmod(16, mulmod(u3, v, n), n);
    let denom_inv = match mod_inverse(denom, n) {
        Some(inv) => inv,
        None => {
            // gcd(denom, n) > 1 — this IS a factor!
            let g = gcd(denom, n);
            if g > 1 && g < n {
                return Some(g);
            }
            return None;
        }
    };

    let a24_normal = mulmod(numerator, denom_inv, n);
    let curve = CurveParams {
        a24: mp.to_mont(a24_normal),
    };

    let mut q = Point {
        x: mp.to_mont(u3),
        z: mp.to_mont(v3),
    };

    // --- Stage 1 ---
    // We accumulate a product of Z-coordinates and check gcd periodically
    // to detect factors early and avoid the "overshot" case (gcd == n)
    // that happens when B1 is large relative to the factors.
    let mut accum = mp.r_mod_n; // 1 in Montgomery form
    let mut step_count = 0u32;

    for &p in primes {
        if p > b1 {
            break;
        }
        let mut pk = p;
        while pk <= b1 {
            q = scalar_mul(q, p, &curve, &mp);
            pk = pk.saturating_mul(p);
        }

        // Accumulate Z into product for batched gcd.
        accum = mp.mul(accum, q.z);
        step_count += 1;

        // Check gcd every 16 primes.
        if step_count % 16 == 0 {
            let accum_normal = mp.from_mont(accum);
            let g = gcd(accum_normal, n);
            if g > 1 && g < n {
                return Some(g);
            }
            if g == n {
                // Overshot in this batch — retry per-prime gcd for the
                // last 16 primes.  Restart from the beginning with per-prime
                // checks (acceptable cost for u64 n).
                return ecm_one_curve_careful(n, b1, b2, sigma, primes);
            }
        }
    }

    // Final stage 1 gcd check.
    let accum_normal = mp.from_mont(accum);
    let g = gcd(accum_normal, n);
    if g > 1 && g < n {
        return Some(g);
    }
    if g == n {
        return ecm_one_curve_careful(n, b1, b2, sigma, primes);
    }

    // --- Stage 2 (simple sequential) ---
    // For each prime q in (B1, B2], compute [q]Q and check Z.
    // We use the baby-step approach: precompute Q, 2Q, then step through
    // differences.  For simplicity (and since our n < 2^64), a direct
    // scalar multiply per prime is acceptable.
    //
    // Accumulate a product of Z-values and take batched GCDs.
    let mut accum = mp.r_mod_n; // 1 in Montgomery form

    const BATCH: usize = 32;
    let stage2_primes: Vec<u64> = primes
        .iter()
        .copied()
        .filter(|&p| p > b1 && p <= b2)
        .collect();

    for chunk in stage2_primes.chunks(BATCH) {
        for &p in chunk {
            let qp = scalar_mul(q, p, &curve, &mp);
            accum = mp.mul(accum, qp.z);
        }

        let accum_normal = mp.from_mont(accum);
        let g = gcd(accum_normal, n);
        if g > 1 && g < n {
            return Some(g);
        }
        if g == n {
            // Overshot — retry per-prime.
            let mut acc2 = mp.r_mod_n;
            for &p in chunk {
                let qp = scalar_mul(q, p, &curve, &mp);
                acc2 = mp.mul(acc2, qp.z);
                let a2 = mp.from_mont(acc2);
                let g2 = gcd(a2, n);
                if g2 > 1 && g2 < n {
                    return Some(g2);
                }
            }
            return None;
        }
    }

    None
}

/// Fallback ECM that checks gcd after every single prime in Stage 1.
///
/// Called when the batched Stage 1 overshoots (gcd == n), meaning both
/// factors' group orders are multiples of M.  By checking per-prime we
/// can catch the factor before the other one also becomes zero.
fn ecm_one_curve_careful(n: u64, b1: u64, b2: u64, sigma: u64, primes: &[u64]) -> Option<u64> {
    if n <= 1 || n % 2 == 0 {
        return None;
    }

    let mp = MontgomeryParams::new(n);

    let sigma_m = sigma % n;
    let u = submod(mulmod(sigma_m, sigma_m, n), 5, n);
    let v = mulmod(4, sigma_m, n);
    if u == 0 || v == 0 {
        return None;
    }

    let u3 = mulmod(mulmod(u, u, n), u, n);
    let v3 = mulmod(mulmod(v, v, n), v, n);

    let v_minus_u = submod(v, u, n);
    let vmu3 = mulmod(mulmod(v_minus_u, v_minus_u, n), v_minus_u, n);
    let three_u_plus_v = addmod(mulmod(3, u, n), v, n);
    let numerator = mulmod(vmu3, three_u_plus_v, n);
    let denom = mulmod(16, mulmod(u3, v, n), n);
    let denom_inv = match mod_inverse(denom, n) {
        Some(inv) => inv,
        None => {
            let g = gcd(denom, n);
            if g > 1 && g < n {
                return Some(g);
            }
            return None;
        }
    };

    let a24_normal = mulmod(numerator, denom_inv, n);
    let curve = CurveParams {
        a24: mp.to_mont(a24_normal),
    };
    let mut q = Point {
        x: mp.to_mont(u3),
        z: mp.to_mont(v3),
    };

    // Stage 1 with per-prime gcd checks.
    for &p in primes {
        if p > b1 {
            break;
        }
        let mut pk = p;
        while pk <= b1 {
            q = scalar_mul(q, p, &curve, &mp);
            pk = pk.saturating_mul(p);
        }
        let z_normal = mp.from_mont(q.z);
        let g = gcd(z_normal, n);
        if g > 1 && g < n {
            return Some(g);
        }
        if g == n {
            return None; // Both factors' orders divide M at this prime — give up.
        }
    }

    // Stage 2 with per-prime checks.
    for &p in primes {
        if p <= b1 {
            continue;
        }
        if p > b2 {
            break;
        }
        let qp = scalar_mul(q, p, &curve, &mp);
        let z_normal = mp.from_mont(qp.z);
        let g = gcd(z_normal, n);
        if g > 1 && g < n {
            return Some(g);
        }
    }

    None
}

/// ECM B1/B2 bounds sequence (CADO-NFS-inspired).
///
/// Returns a list of `(B1, B2)` pairs to try sequentially.  For small
/// `lpb` values (< 20) no ECM curves are attempted.
pub fn ecm_bounds(lpb: u32) -> Vec<(u64, u64)> {
    let ncurves = match lpb {
        0..=19 => 0,
        20..=22 => 1,
        23 => 2,
        24 => 4,
        25 => 5,
        26 => 6,
        27 => 8,
        28 => 11,
        _ => 16,
    };

    let mut bounds = Vec::with_capacity(ncurves);
    let mut b1 = 105.0f64;
    for _ in 0..ncurves {
        let b2 = ((2.0 * (50.0 * b1 / 210.0).floor() + 1.0) * 105.0) as u64;
        bounds.push((b1 as u64, b2));
        b1 += b1.sqrt();
    }
    bounds
}

// ---------------------------------------------------------------------------
// Montgomery-curve point operations
// ---------------------------------------------------------------------------

/// Scalar multiplication `[k]P` on a Montgomery curve using the
/// Montgomery ladder (constant-time w.r.t. bit pattern, and only needs
/// differential addition).
fn scalar_mul(p: Point, k: u64, curve: &CurveParams, mp: &MontgomeryParams) -> Point {
    if k == 0 {
        // Point at infinity represented as (0 : 0) — but for ECM this
        // should not happen with proper bounds.
        return Point {
            x: mp.r_mod_n,
            z: 0,
        };
    }
    if k == 1 {
        return p;
    }

    // Montgomery ladder.
    let mut r0 = p; // R0 = P
    let mut r1 = xdbl(p, curve, mp); // R1 = [2]P

    let bits = 64 - k.leading_zeros();
    for i in (0..bits - 1).rev() {
        if (k >> i) & 1 == 1 {
            r0 = xadd(r0, r1, p, mp);
            r1 = xdbl(r1, curve, mp);
        } else {
            r1 = xadd(r0, r1, p, mp);
            r0 = xdbl(r0, curve, mp);
        }
    }

    r0
}

/// Montgomery curve point doubling: `(X2, Z2) = [2](X, Z)`.
///
/// Uses the standard formulas with `a24 = (A + 2) / 4`:
/// ```text
/// u = (X + Z)^2,  v = (X - Z)^2,  diff = u - v
/// X2 = u * v,     Z2 = diff * (v + a24 * diff)
/// ```
/// All arithmetic in Montgomery form.
#[inline]
fn xdbl(p: Point, curve: &CurveParams, mp: &MontgomeryParams) -> Point {
    let sum = addmod_mont(p.x, p.z, mp.n);
    let diff = submod_mont(p.x, p.z, mp.n);
    let u = mp.sqr(sum); // (X + Z)^2
    let v = mp.sqr(diff); // (X - Z)^2
    let d = submod_mont(u, v, mp.n); // u - v
    let x2 = mp.mul(u, v);
    let t = mp.mul(curve.a24, d);
    let z2 = mp.mul(d, addmod_mont(v, t, mp.n));
    Point { x: x2, z: z2 }
}

/// Montgomery curve differential addition:
/// Given `P1`, `P2`, and `P0 = P1 - P2`, compute `P1 + P2`.
///
/// ```text
/// u1 = (X1 - Z1) * (X2 + Z2)
/// u2 = (X1 + Z1) * (X2 - Z2)
/// X3 = Z0 * (u1 + u2)^2
/// Z3 = X0 * (u1 - u2)^2
/// ```
/// All arithmetic in Montgomery form.
#[inline]
fn xadd(p1: Point, p2: Point, p0: Point, mp: &MontgomeryParams) -> Point {
    let u1 = mp.mul(submod_mont(p1.x, p1.z, mp.n), addmod_mont(p2.x, p2.z, mp.n));
    let u2 = mp.mul(addmod_mont(p1.x, p1.z, mp.n), submod_mont(p2.x, p2.z, mp.n));
    let sum = addmod_mont(u1, u2, mp.n);
    let diff = submod_mont(u1, u2, mp.n);
    let x3 = mp.mul(p0.z, mp.sqr(sum));
    let z3 = mp.mul(p0.x, mp.sqr(diff));
    Point { x: x3, z: z3 }
}

// ---------------------------------------------------------------------------
// Arithmetic helpers
// ---------------------------------------------------------------------------

/// `(a + b) mod n` for values in Montgomery form (both < n).
#[inline]
fn addmod_mont(a: u64, b: u64, n: u64) -> u64 {
    let s = a as u128 + b as u128;
    (s % n as u128) as u64
}

/// `(a - b) mod n` for values in Montgomery form (both < n).
#[inline]
fn submod_mont(a: u64, b: u64, n: u64) -> u64 {
    if a >= b {
        a - b
    } else {
        // a + n - b; since both < n, result is in [1, n-1].
        n - b + a
    }
}

/// `(a * b) mod n` using u128 intermediate.
#[inline]
fn mulmod(a: u64, b: u64, n: u64) -> u64 {
    ((a as u128 * b as u128) % n as u128) as u64
}

/// `(a - b) mod n`, handling underflow.
#[inline]
fn submod(a: u64, b: u64, n: u64) -> u64 {
    if a >= b {
        (a - b) % n
    } else {
        ((a as u128 + n as u128 - b as u128) % n as u128) as u64
    }
}

/// `(a + b) mod n`.
#[inline]
fn addmod(a: u64, b: u64, n: u64) -> u64 {
    ((a as u128 + b as u128) % n as u128) as u64
}

/// Binary GCD.
fn gcd(a: u64, b: u64) -> u64 {
    let (mut a, mut b) = (a, b);
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ecm_bounds_empty_for_small_lpb() {
        assert!(ecm_bounds(17).is_empty());
        assert!(ecm_bounds(18).is_empty());
        assert!(ecm_bounds(19).is_empty());
    }

    #[test]
    fn test_ecm_bounds_nonempty_for_large_lpb() {
        let bounds = ecm_bounds(20);
        assert_eq!(bounds.len(), 1);
        let bounds = ecm_bounds(24);
        assert_eq!(bounds.len(), 4);
        let bounds = ecm_bounds(28);
        assert_eq!(bounds.len(), 11);
    }

    #[test]
    fn test_ecm_bounds_b2_gt_b1() {
        for lpb in 20..=30 {
            for &(b1, b2) in &ecm_bounds(lpb) {
                assert!(b2 > b1, "b2={} should be > b1={} for lpb={}", b2, b1, lpb);
            }
        }
    }

    #[test]
    fn test_xdbl_basic() {
        // Verify that doubling doesn't panic and produces valid output.
        let n = 1009u64 * 1013; // 1022117
        let mp = MontgomeryParams::new(n);
        let curve = CurveParams { a24: mp.to_mont(3) };
        let p = Point {
            x: mp.to_mont(7),
            z: mp.to_mont(1),
        };
        let p2 = xdbl(p, &curve, &mp);
        // Just verify no panic and Z != 0 (not point at infinity).
        assert!(p2.z != 0 || p2.x != 0);
    }

    #[test]
    fn test_ecm_one_curve_finds_factor() {
        // n = 1009 * 1013 = 1022117
        // Try several sigma values; at least one should find a factor.
        let n = 1009u64 * 1013;
        let mut found = false;
        for sigma in 3..20u64 {
            if let Some(f) = ecm_one_curve(n, 500, 5000, sigma) {
                assert!(n % f == 0, "factor {} does not divide {}", f, n);
                assert!(f > 1 && f < n, "trivial factor {}", f);
                found = true;
                break;
            }
        }
        assert!(
            found,
            "ECM should find a factor of 1009*1013 with some sigma"
        );
    }

    #[test]
    fn test_ecm_one_curve_prime_returns_none() {
        // A prime should never yield a non-trivial factor.
        for sigma in 3..10u64 {
            assert!(ecm_one_curve(1009, 500, 5000, sigma).is_none());
        }
    }

    #[test]
    fn test_ecm_one_curve_even_returns_none() {
        assert!(ecm_one_curve(100, 500, 5000, 6).is_none());
    }

    #[test]
    fn test_ecm_larger_semiprime() {
        // n = 10007 * 10009 = 100_160_063
        let n = 10007u64 * 10009;
        let mut found = false;
        for sigma in 3..30u64 {
            if let Some(f) = ecm_one_curve(n, 1000, 10000, sigma) {
                assert!(n % f == 0);
                assert!(f > 1 && f < n);
                found = true;
                break;
            }
        }
        assert!(
            found,
            "ECM should find a factor of 10007*10009 with some sigma"
        );
    }
}
