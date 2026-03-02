//! Q-lattice reduction for the special-q lattice sieve.
//!
//! The special-q lattice sieve works in a 2D (i, j) coordinate system where
//! (a, b) = (a0*i + a1*j, b0*i + b1*j). The q-lattice basis {(a0,b0),(a1,b1)}
//! is a reduced basis for the lattice of (a,b) pairs with a ≡ r*b (mod q),
//! where (q, r) is the special-q.

/// Reduced basis for the q-lattice.
///
/// The lattice is { (a,b) in Z^2 : a ≡ r*b (mod q) }.
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
/// Given special-q prime `q` with root `r` (i.e., f(r) ≡ 0 mod q) and a
/// skewness parameter, returns a reduced basis {v0, v1} such that |det| = q
/// and both vectors are short with respect to the skewed norm
/// Q(a, b) = a^2 + S^2 * b^2.
///
/// Start with the generating vectors v0 = (q, 0) and v1 = (r, 1).
/// Iteratively subtract the nearest integer multiple of the shorter vector
/// from the longer one (Lagrange / Gauss reduction with skewed inner product).
pub fn reduce_qlattice(q: u64, r: u64, skewness: f64) -> QLattice {
    assert!(q > 0, "reduce_qlattice: q must be > 0");
    assert!(r < q, "reduce_qlattice: r must be < q");

    let s2 = skewness * skewness;

    let mut a0 = q as i64;
    let mut b0: i64 = 0;
    let mut a1 = r as i64;
    let mut b1: i64 = 1;

    let qform = |a: i64, b: i64| -> f64 {
        (a as f64) * (a as f64) + s2 * (b as f64) * (b as f64)
    };

    let inner = |xa: i64, xb: i64, ya: i64, yb: i64| -> f64 {
        (xa as f64) * (ya as f64) + s2 * (xb as f64) * (yb as f64)
    };

    loop {
        let q0 = qform(a0, b0);
        let q1 = qform(a1, b1);

        if q0 > q1 {
            std::mem::swap(&mut a0, &mut a1);
            std::mem::swap(&mut b0, &mut b1);
        }

        let q0 = qform(a0, b0);
        let q1_before = qform(a1, b1);

        let dot = inner(a0, b0, a1, b1);
        let mu = (dot / q0).round() as i64;

        if mu == 0 {
            break;
        }

        a1 -= mu * a0;
        b1 -= mu * b0;

        let q1_after = qform(a1, b1);
        if q1_after >= q1_before {
            break;
        }
    }

    QLattice { a0, b0, a1, b1 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qlattice_basic() {
        let ql = reduce_qlattice(97, 30, 1.0);
        let det = (ql.a0 as i128 * ql.b1 as i128 - ql.a1 as i128 * ql.b0 as i128).abs();
        assert_eq!(det, 97, "determinant should be q=97");
    }

    #[test]
    fn test_qlattice_short_vectors() {
        let ql = reduce_qlattice(65537, 12345, 1.0);
        let det = (ql.a0 as i128 * ql.b1 as i128 - ql.a1 as i128 * ql.b0 as i128).abs();
        assert_eq!(det, 65537);
        let len0 = ((ql.a0 as f64).powi(2) + (ql.b0 as f64).powi(2)).sqrt();
        let len1 = ((ql.a1 as f64).powi(2) + (ql.b1 as f64).powi(2)).sqrt();
        assert!(len0 < 512.0, "v0 too long: {}", len0);
        assert!(len1 < 512.0, "v1 too long: {}", len1);
    }

    #[test]
    fn test_qlattice_determinant_preserved() {
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
        let q = 97u64;
        let r = 30u64;
        let ql = reduce_qlattice(q, r, 1.0);

        let check = |a: i64, b: i64| {
            let lhs = ((a as i128) % (q as i128) + (q as i128)) % (q as i128);
            let rhs = (((r as i128) * (b as i128)) % (q as i128) + (q as i128)) % (q as i128);
            assert_eq!(lhs, rhs, "({}, {}) not in q-lattice: {} != {}", a, b, lhs, rhs);
        };
        check(ql.a0, ql.b0);
        check(ql.a1, ql.b1);
    }

    #[test]
    fn test_qlattice_with_skew() {
        let ql = reduce_qlattice(65537, 12345, 2.0);
        let det = (ql.a0 as i128 * ql.b1 as i128 - ql.a1 as i128 * ql.b0 as i128).abs();
        assert_eq!(det, 65537);
    }

    #[test]
    fn test_qlattice_r_zero() {
        let ql = reduce_qlattice(97, 0, 1.0);
        let det = (ql.a0 as i128 * ql.b1 as i128 - ql.a1 as i128 * ql.b0 as i128).abs();
        assert_eq!(det, 97);
    }

    #[test]
    fn test_qlattice_r_one() {
        let ql = reduce_qlattice(97, 1, 1.0);
        let det = (ql.a0 as i128 * ql.b1 as i128 - ql.a1 as i128 * ql.b0 as i128).abs();
        assert_eq!(det, 97);

        let check = |a: i64, b: i64| {
            let lhs = ((a as i128) % 97 + 97) % 97;
            let rhs = ((b as i128) % 97 + 97) % 97;
            assert_eq!(lhs, rhs, "({}, {}) not in q-lattice: {} != {}", a, b, lhs, rhs);
        };
        check(ql.a0, ql.b0);
        check(ql.a1, ql.b1);
    }
}
