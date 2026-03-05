//! Q-lattice and P-lattice reduction for the special-q lattice sieve.
//!
//! The special-q lattice sieve works in a 2D (i, j) coordinate system where
//! (a, b) = (a0*i + a1*j, b0*i + b1*j). The q-lattice basis {(a0,b0),(a1,b1)}
//! is a reduced basis for the lattice of (a,b) pairs with a - r*b === 0 (mod q),
//! where (q, r) is the special-q.
//!
//! For each factor base prime p with root R, we need to enumerate all (i,j) in
//! the sieve region where p | F(a,b). This requires transforming the root R
//! through the q-lattice and performing a partial-GCD reduction to obtain short
//! walk vectors (the Franke-Kleinjung enumeration).

use crate::arith::mod_inverse;

// ---------------------------------------------------------------------------
// Q-Lattice: reduced basis for the special-q sublattice
// ---------------------------------------------------------------------------

/// Reduced basis for the q-lattice.
///
/// The lattice is { (a,b) in Z^2 : a === r*b (mod q) }.
/// After Gaussian reduction, {(a0,b0), (a1,b1)} is a short basis with
/// |det| = q.
#[derive(Debug, Clone, Copy)]
pub struct QLattice {
    pub a0: i64,
    pub b0: i64,
    pub a1: i64,
    pub b1: i64,
}

/// Perform skew Gaussian lattice reduction on the q-lattice.
///
/// Given special-q prime `q` with root `r` (i.e., f(r) === 0 mod q) and a
/// skewness parameter, returns a reduced basis {v0, v1} such that |det| = q
/// and both vectors are short with respect to the skewed norm
/// Q(a, b) = a^2 + S^2 * b^2.
///
/// # Algorithm
///
/// Start with the generating vectors v0 = (q, 0) and v1 = (r, 1).
/// Iteratively subtract the nearest integer multiple of the shorter vector
/// from the longer one (Lagrange / Gauss reduction with skewed inner product).
///
/// # Panics
///
/// Panics if `q == 0` or `r >= q`.
pub fn reduce_qlattice(q: u64, r: u64, skewness: f64) -> QLattice {
    assert!(q > 0, "reduce_qlattice: q must be > 0");
    assert!(r < q, "reduce_qlattice: r must be < q");

    let s2 = skewness * skewness;

    // Skewed quadratic form: Q(a, b) = a^2 + S^2 * b^2
    // We work in i64 to allow negative coordinates.
    let mut a0 = q as i64;
    let mut b0: i64 = 0;
    let mut a1 = r as i64;
    let mut b1: i64 = 1;

    // Q(v) using f64 to avoid overflow for large q
    let qform = |a: i64, b: i64| -> f64 { (a as f64) * (a as f64) + s2 * (b as f64) * (b as f64) };

    // Skewed inner product: <v0, v1> = a0*a1 + S^2 * b0*b1
    let inner = |xa: i64, xb: i64, ya: i64, yb: i64| -> f64 {
        (xa as f64) * (ya as f64) + s2 * (xb as f64) * (yb as f64)
    };

    loop {
        let q0 = qform(a0, b0);
        let q1 = qform(a1, b1);

        // Ensure v0 is the shorter vector
        if q0 > q1 {
            std::mem::swap(&mut a0, &mut a1);
            std::mem::swap(&mut b0, &mut b1);
        }

        let q0 = qform(a0, b0);
        let q1_before = qform(a1, b1);

        // mu = round(<v0, v1> / Q(v0))
        let dot = inner(a0, b0, a1, b1);
        let mu = (dot / q0).round() as i64;

        if mu == 0 {
            break;
        }

        // v1 = v1 - mu * v0
        a1 -= mu * a0;
        b1 -= mu * b0;

        // Guard against oscillation: if Q(v1) did not strictly decrease,
        // the basis is already Gauss-reduced and we should stop.
        let q1_after = qform(a1, b1);
        if q1_after >= q1_before {
            break;
        }
    }

    QLattice { a0, b0, a1, b1 }
}

// ---------------------------------------------------------------------------
// P-Lattice: Franke-Kleinjung enumeration parameters
// ---------------------------------------------------------------------------

/// Parameters for enumerating hits of a factor base prime p within the sieve
/// region, using the Franke-Kleinjung walk pattern.
///
/// The walk traverses a 1D sieve array. Starting from `start`, we step through
/// the array using two increments (`inc_step` and `inc_warp`) with bounds that
/// control when to switch between them.
#[derive(Debug, Clone, Copy)]
pub struct PLattice {
    /// Starting offset in the sieve line (first hit in row j=0, if any).
    pub start: u64,
    /// Primary step increment (1D encoding of one reduced basis vector).
    pub inc_step: i64,
    /// Secondary warp increment (1D encoding of the other reduced basis vector).
    pub inc_warp: i64,
    /// Bound for the step direction.
    pub bound_step: i64,
    /// Bound for the warp direction.
    pub bound_warp: i64,
    /// Whether this prime actually hits the sieve region.
    pub hits: bool,
}

/// Reduce the p-lattice for a factor base prime `p` with polynomial root `r`,
/// given the current q-lattice basis and sieve half-width `log_i` (the sieve
/// region spans i in [-I, I) where I = 1 << log_i).
///
/// # Algorithm
///
/// 1. Transform root R through the q-lattice to get R' in (i,j) space:
///    R' = (R*b1 - a1) * (a0 - R*b0)^{-1} mod p
///
/// 2. Start with basis vectors (p, 0) and (R', 1) for the sublattice of
///    (i,j) pairs where p divides the algebraic norm.
///
/// 3. Perform partial-GCD reduction (truncated extended Euclidean algorithm)
///    until the first coordinate is shorter than I = 1 << log_i.
///
/// 4. Extract the FK walk parameters from the reduced basis.
///
/// Returns a `PLattice` with `hits = false` if the modular inverse does not
/// exist (p divides a q-lattice coefficient) or the prime is too large.
pub fn reduce_plattice(p: u64, r: u64, qlat: &QLattice, log_i: u32) -> PLattice {
    let no_hit = PLattice {
        start: 0,
        inc_step: 0,
        inc_warp: 0,
        bound_step: 0,
        bound_warp: 0,
        hits: false,
    };

    if p <= 1 {
        return no_hit;
    }

    let half_width = 1i64 << log_i; // I = sieve half-width

    // Step 1: Transform root R from (a,b)-space to (i,j)-space.
    //
    // In the q-lattice, (a,b) = (a0*i + a1*j, b0*i + b1*j).
    // We need a === R*b (mod p), i.e.:
    //   (a0*i + a1*j) === R*(b0*i + b1*j) (mod p)
    //   i*(a0 - R*b0) === j*(R*b1 - a1) (mod p)
    //
    // So the transformed root is:
    //   R' = (R*b1 - a1) * (a0 - R*b0)^{-1} mod p

    let p_i128 = p as i128;

    // Compute (a0 - R*b0) mod p, carefully handling signs
    let denom = ((qlat.a0 as i128 - (r as i128) * (qlat.b0 as i128)) % p_i128 + p_i128) % p_i128;
    let numer = (((r as i128) * (qlat.b1 as i128) - qlat.a1 as i128) % p_i128 + p_i128) % p_i128;

    if denom == 0 {
        // p divides (a0 - R*b0); the transformed root is "infinity" in projective
        // coordinates, meaning all hits are in column i=0. For the sieve we skip
        // these (they correspond to the j-line, handled separately if needed).
        // However, check the numerator case: if both are 0, every (i,j) is a hit
        // modulo p (projective root). Still skip for the FK walk.
        return no_hit;
    }

    let denom_u64 = denom as u64;
    let inv = match mod_inverse(denom_u64, p) {
        Some(v) => v,
        None => return no_hit,
    };

    let r_prime = ((numer as u128 * inv as u128) % p as u128) as u64;

    // Step 2: Partial-GCD reduction.
    //
    // Basis: (u0, v0) = (p, 0), (u1, v1) = (r', 1)
    // Reduce until |u1| < I (half_width).

    let mut u0 = p as i64;
    let mut v0: i64 = 0;
    let mut u1 = r_prime as i64;
    let mut v1: i64 = 1;

    // Handle the case where r_prime >= p/2 by centering
    if u1 > (p as i64) / 2 {
        u1 -= p as i64;
        // v1 stays 1 (conceptually we took (r' - p, 1) which is equivalent mod p)
    }

    // Run the partial-GCD (truncated extended Euclidean)
    while u1 != 0 && u1.unsigned_abs() >= half_width as u64 {
        let q_div = u0 / u1;
        let new_u = u0 - q_div * u1;
        let new_v = v0 - q_div * v1;
        u0 = u1;
        v0 = v1;
        u1 = new_u;
        v1 = new_v;
    }

    // After reduction, we have two basis vectors in (i, j) space:
    //   vec_a = (u0, v0) and vec_b = (u1, v1)
    // with |u1| < I and |u0| >= I (or u1 == 0 if p < I).
    //
    // For the FK walk, we need:
    // - The vector with |u| < I gives the "step" direction
    // - The vector with |u| >= I gives the "warp" direction

    if u1 == 0 {
        // Degenerate: p divides the sieve width or r_prime = 0.
        // All hits are at i = 0 mod p. If p < I, step by p.
        if (p as i64) < half_width {
            return PLattice {
                start: 0,
                inc_step: p as i64,
                inc_warp: 0,
                bound_step: half_width,
                bound_warp: 0,
                hits: true,
            };
        }
        return no_hit;
    }

    // Identify which vector is "step" (short u) and which is "warp" (long u).
    let (su, sv, wu, wv) = if u1.unsigned_abs() < u0.unsigned_abs() {
        (u1, v1, u0, v0)
    } else {
        (u0, v0, u1, v1)
    };

    // Ensure step vector has positive j-component (canonical form)
    let (su, sv) = if sv < 0 { (-su, -sv) } else { (su, sv) };
    let (wu, wv) = if wv < 0 { (-wu, -wv) } else { (wu, wv) };

    // The FK walk parameters:
    // inc_step encodes: advance by (su, sv) in the (i, j) sieve grid.
    // In a 1D sieve array of width 2*I, position = j * (2*I) + (i + I).
    // So inc_step = sv * (2*I) + su, and similarly for inc_warp.
    let sieve_width = 2 * half_width;
    let inc_step = sv * sieve_width + su;
    let inc_warp = wv * sieve_width + wu;

    // Bounds: the step can be applied while i stays in [-I, I).
    // bound_step = I - |su| (roughly: how many steps before we need a warp).
    let bound_step = half_width - su.abs();
    let bound_warp = half_width - wu.abs().min(half_width);

    // Compute start position: first hit in the sieve region.
    // For j=0: need i such that i === 0 (mod gcd(basis)), which for prime p
    // means the first hit in [-I, I) is at i = 0 if p < I, else we need to
    // search. For simplicity, start at 0 (the FK walk will find all hits).
    let start = 0u64;

    PLattice {
        start,
        inc_step,
        inc_warp,
        bound_step,
        bound_warp,
        hits: true,
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qlattice_basic() {
        let ql = reduce_qlattice(97, 30, 1.0);
        // Verify determinant = +/- q
        let det = (ql.a0 as i128 * ql.b1 as i128 - ql.a1 as i128 * ql.b0 as i128).abs();
        assert_eq!(det, 97, "determinant should be q=97");
    }

    #[test]
    fn test_qlattice_short_vectors() {
        let ql = reduce_qlattice(65537, 12345, 1.0);
        let det = (ql.a0 as i128 * ql.b1 as i128 - ql.a1 as i128 * ql.b0 as i128).abs();
        assert_eq!(det, 65537);
        // Both vectors should be short (< 2*sqrt(q) ~ 512)
        let len0 = ((ql.a0 as f64).powi(2) + (ql.b0 as f64).powi(2)).sqrt();
        let len1 = ((ql.a1 as f64).powi(2) + (ql.b1 as f64).powi(2)).sqrt();
        assert!(len0 < 512.0, "v0 too long: {}", len0);
        assert!(len1 < 512.0, "v1 too long: {}", len1);
    }

    #[test]
    fn test_qlattice_with_skew() {
        let ql = reduce_qlattice(65537, 12345, 2.0);
        let det = (ql.a0 as i128 * ql.b1 as i128 - ql.a1 as i128 * ql.b0 as i128).abs();
        assert_eq!(det, 65537);
    }

    #[test]
    fn test_qlattice_determinant_preserved() {
        // Test across a range of (q, r) pairs that the determinant is always q.
        let test_cases: Vec<(u64, u64)> = vec![
            (7, 3),
            (13, 5),
            (101, 42),
            (1009, 500),
            (65521, 33333),
            (104729, 12345),
        ];
        for (q, r) in test_cases {
            let ql = reduce_qlattice(q, r, 1.0);
            let det = (ql.a0 as i128 * ql.b1 as i128 - ql.a1 as i128 * ql.b0 as i128).abs();
            assert_eq!(det, q as i128, "det should be q={} for r={}", q, r);
        }
    }

    #[test]
    fn test_qlattice_generates_lattice_point() {
        // Every (a, b) in the q-lattice should satisfy a === r*b (mod q).
        let q = 97u64;
        let r = 30u64;
        let ql = reduce_qlattice(q, r, 1.0);

        // Check both basis vectors: a === r*b (mod q)
        let check = |a: i64, b: i64| {
            let lhs = ((a as i128) % (q as i128) + (q as i128)) % (q as i128);
            let rhs = (((r as i128) * (b as i128)) % (q as i128) + (q as i128)) % (q as i128);
            assert_eq!(
                lhs, rhs,
                "({}, {}) not in q-lattice: {} != {}",
                a, b, lhs, rhs
            );
        };
        check(ql.a0, ql.b0);
        check(ql.a1, ql.b1);
    }

    #[test]
    fn test_qlattice_skew_affects_basis() {
        // Different skewness should produce different (but valid) bases.
        let q = 65537u64;
        let r = 12345u64;
        let ql1 = reduce_qlattice(q, r, 1.0);
        let ql2 = reduce_qlattice(q, r, 4.0);

        // Both must have correct determinant
        let det1 = (ql1.a0 as i128 * ql1.b1 as i128 - ql1.a1 as i128 * ql1.b0 as i128).abs();
        let det2 = (ql2.a0 as i128 * ql2.b1 as i128 - ql2.a1 as i128 * ql2.b0 as i128).abs();
        assert_eq!(det1, q as i128);
        assert_eq!(det2, q as i128);

        // With higher skew, the b-components should be relatively smaller
        // (the skewed norm penalizes large b). This is a soft check.
        let b_norm1 = (ql1.b0 as f64).powi(2) + (ql1.b1 as f64).powi(2);
        let b_norm2 = (ql2.b0 as f64).powi(2) + (ql2.b1 as f64).powi(2);
        // With skew=4, the algorithm penalizes b more, so b_norm2 should
        // generally be <= b_norm1. Allow some tolerance.
        assert!(
            b_norm2 <= b_norm1 * 1.5,
            "skew=4 should not increase b-norm much: {} vs {}",
            b_norm2,
            b_norm1
        );
    }

    #[test]
    fn test_qlattice_r_zero() {
        // r = 0 means the lattice is {(a, b) : a === 0 (mod q)} = {(q*k, b)}.
        // Basis should be (q, 0) and (0, 1).
        let ql = reduce_qlattice(97, 0, 1.0);
        let det = (ql.a0 as i128 * ql.b1 as i128 - ql.a1 as i128 * ql.b0 as i128).abs();
        assert_eq!(det, 97);
    }

    #[test]
    fn test_qlattice_r_one() {
        let ql = reduce_qlattice(97, 1, 1.0);
        let det = (ql.a0 as i128 * ql.b1 as i128 - ql.a1 as i128 * ql.b0 as i128).abs();
        assert_eq!(det, 97);

        // Both basis vectors should satisfy a === b (mod 97)
        let check = |a: i64, b: i64| {
            let lhs = ((a as i128) % 97 + 97) % 97;
            let rhs = ((b as i128) % 97 + 97) % 97;
            assert_eq!(
                lhs, rhs,
                "({}, {}) not in q-lattice: {} != {}",
                a, b, lhs, rhs
            );
        };
        check(ql.a0, ql.b0);
        check(ql.a1, ql.b1);
    }

    #[test]
    fn test_plattice_small_prime() {
        let ql = reduce_qlattice(97, 30, 1.0);
        let pl = reduce_plattice(7, 3, &ql, 9);
        assert!(pl.hits, "prime 7 should hit sieve region");
    }

    #[test]
    fn test_plattice_large_prime() {
        let ql = reduce_qlattice(65537, 12345, 1.0);
        let pl = reduce_plattice(521, 100, &ql, 9);
        // Large primes should still have reasonable walk parameters
        if pl.hits {
            assert!(pl.inc_step != 0 || pl.inc_warp != 0);
        }
    }

    #[test]
    fn test_plattice_root_zero() {
        // r = 0: the algebraic norm is divisible by p when a === 0 (mod p),
        // i.e., a0*i + a1*j === 0 (mod p).
        let ql = reduce_qlattice(97, 30, 1.0);
        let pl = reduce_plattice(7, 0, &ql, 9);
        // Should produce valid walk parameters (or no-hit if degenerate)
        // Just verify no panic.
        let _ = pl;
    }

    #[test]
    fn test_plattice_prime_2() {
        // p = 2 is a valid factor base prime; r can only be 0 or 1.
        let ql = reduce_qlattice(97, 30, 1.0);
        let pl = reduce_plattice(2, 1, &ql, 9);
        // p=2 is very small relative to I=512, so it should hit.
        // The walk parameters might be trivial but should not panic.
        let _ = pl;
    }

    #[test]
    fn test_plattice_p_equals_1() {
        // p = 1 is degenerate and should return no-hit.
        let ql = reduce_qlattice(97, 30, 1.0);
        let pl = reduce_plattice(1, 0, &ql, 9);
        assert!(!pl.hits, "p=1 should not hit");
    }

    #[test]
    fn test_plattice_various_primes() {
        // Test a range of factor base primes with various roots.
        let ql = reduce_qlattice(65537, 12345, 1.0);
        let test_cases: Vec<(u64, u64)> = vec![
            (3, 1),
            (5, 2),
            (11, 7),
            (31, 15),
            (127, 50),
            (257, 100),
            (1021, 500),
        ];
        for (p, r) in test_cases {
            let pl = reduce_plattice(p, r, &ql, 10);
            // All small primes (relative to sieve width 2^10 = 1024) should hit.
            if p < (1 << 10) {
                assert!(
                    pl.hits,
                    "prime {} with root {} should hit sieve region",
                    p, r
                );
            }
            if pl.hits {
                assert!(
                    pl.inc_step != 0 || pl.inc_warp != 0,
                    "prime {} must have nonzero walk increment",
                    p
                );
            }
        }
    }

    #[test]
    fn test_plattice_with_skewed_qlattice() {
        let ql = reduce_qlattice(65537, 12345, 3.0);
        let pl = reduce_plattice(31, 10, &ql, 10);
        assert!(pl.hits, "prime 31 should hit with skewed q-lattice");
    }

    #[test]
    fn test_plattice_large_sieve_region() {
        // log_i = 15 means I = 32768, so most FB primes should hit easily.
        let ql = reduce_qlattice(65537, 12345, 1.0);
        let pl = reduce_plattice(521, 100, &ql, 15);
        assert!(pl.hits, "prime 521 should hit with log_i=15");
        assert!(pl.inc_step != 0 || pl.inc_warp != 0);
    }
}
