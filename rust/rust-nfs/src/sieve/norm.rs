//! Norm initialization: fill sieve arrays with u8 log-norm approximations.
//!
//! Before sieving, each cell is initialized with `log2(|norm(i,j)|) * scale`,
//! representing how much "work" the sieve must do to prove this position is smooth.
//! Primes hitting a position subtract their log contributions; survivors are positions
//! where the residual is below a threshold.
//!
//! In the q-lattice, position `(i, j)` maps to `(a, b) = (a0*i + a1*j, b0*i + b1*j)`.

/// Initialize rational sieve array for row `j` of the q-lattice.
///
/// The rational polynomial is `F_rat(a, b) = g1*a + g0*b` (degree 1, homogeneous).
/// In q-lattice coordinates this becomes:
///
/// ```text
/// F_rat(i, j) = g1*(a0*i + a1*j) + g0*(b0*i + b1*j)
///             = (g1*a0 + g0*b0)*i + (g1*a1 + g0*b1)*j
///             = slope * i + intercept
/// ```
///
/// The sieve array covers `i` in `[-half_i, half_i)`. Index `k` corresponds
/// to `i = k - half_i`.
///
/// Each cell is set to `min(255, max(0, floor(log2(|F_rat|) * scale)))` as `u8`.
/// When the norm is zero or less than 1, the cell is set to 0.
pub fn init_norm_rat(
    sieve: &mut [u8],
    g0: f64,
    g1: f64,
    a0: f64,
    b0: f64,
    a1: f64,
    b1: f64,
    j: i32,
    half_i: i32,
    scale: f64,
) {
    let slope = g1 * a0 + g0 * b0;
    let intercept = (g1 * a1 + g0 * b1) * (j as f64);

    for k in 0..sieve.len() {
        let i = (k as i32) - half_i;
        let f_val = slope * (i as f64) + intercept;
        let abs_f = f_val.abs();
        if abs_f < 1.0 {
            sieve[k] = 0;
        } else {
            let log_val = abs_f.log2() * scale;
            sieve[k] = log_val.min(255.0).max(0.0) as u8;
        }
    }
}

/// Initialize algebraic sieve array for row `j` of the q-lattice.
///
/// The algebraic polynomial is the homogeneous form:
///
/// ```text
/// F_alg(a, b) = c_d * a^d + c_{d-1} * a^{d-1} * b + ... + c_0 * b^d
/// ```
///
/// where `f_coeffs = [c_0, c_1, ..., c_d]` (ascending order, matching the crate convention
/// used in `factorbase.rs`). In q-lattice coordinates: `a = a0*i + a1*j`, `b = b0*i + b1*j`.
///
/// Uses Horner evaluation in `f64` for the log approximation. When `b != 0`, we evaluate
/// the univariate polynomial at `t = a/b` and add `d * log2(|b|)`. When `b == 0`, we
/// evaluate using `a` alone.
///
/// Each cell is set to `min(255, max(0, floor(log2(|F_alg|) * scale)))` as `u8`.
pub fn init_norm_alg(
    sieve: &mut [u8],
    f_coeffs: &[i64],
    a0: f64,
    b0: f64,
    a1: f64,
    b1: f64,
    j: i32,
    half_i: i32,
    scale: f64,
) {
    if f_coeffs.is_empty() {
        for v in sieve.iter_mut() {
            *v = 0;
        }
        return;
    }

    let d = f_coeffs.len() - 1; // polynomial degree
    let j_f64 = j as f64;

    for k in 0..sieve.len() {
        let i = (k as i32) - half_i;
        let i_f64 = i as f64;
        let a = a0 * i_f64 + a1 * j_f64;
        let b = b0 * i_f64 + b1 * j_f64;

        let abs_f = if b.abs() > a.abs().max(1.0) * 1e-15 {
            // Horner in t = a/b: poly(t) = c_d * t^d + c_{d-1} * t^{d-1} + ... + c_0
            // F_alg = b^d * poly(t)
            let t = a / b;
            // Evaluate poly(t) using Horner (highest degree first)
            let mut poly = f_coeffs[d] as f64;
            for idx in (0..d).rev() {
                poly = poly * t + f_coeffs[idx] as f64;
            }
            let abs_bd = b.abs().powi(d as i32);
            (poly * abs_bd).abs()
        } else if a.abs() > 1e-30 {
            // b is essentially zero: F_alg = c_d * a^d
            let c_d = f_coeffs[d] as f64;
            (c_d * a.powi(d as i32)).abs()
        } else {
            // Both a and b near zero
            0.0
        };

        if abs_f < 1.0 {
            sieve[k] = 0;
        } else {
            let log_val = abs_f.log2() * scale;
            sieve[k] = log_val.min(255.0).max(0.0) as u8;
        }
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_init_norm_rat_identity_lattice() {
        // Identity q-lattice (a0=1, b0=0, a1=0, b1=1), j=1
        // F_rat = g1*i + g0*1 = i - 10 (g1=1, g0=-10)
        let mut sieve = vec![0u8; 20]; // i in [-10, 10)
        init_norm_rat(&mut sieve, -10.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1, 10, 1.0);
        // At i=10 (index 20-1=19): |10 - 10| = 0 -> log = 0 (or special case)
        // At i=0 (index 10): |0 - 10| = 10 -> log2(10) ~ 3.32 -> 3
        assert!(
            sieve[10] >= 3 && sieve[10] <= 4,
            "norm at i=0 should be ~3.32, got {}",
            sieve[10]
        );
    }

    #[test]
    fn test_init_norm_alg_constant() {
        // f(x) = x^2 + 1, identity lattice, j=0
        // F_alg(i, 0) = i^2 (since b=0, the b^d terms vanish except for a^d)
        // Actually: F(a,b) = a^2 + b^2 for f=[1, 0, 1], j=0 means b=0 so F = a^2 = i^2
        let mut sieve = vec![0u8; 20];
        let coeffs = [1i64, 0, 1]; // x^2 + 1 -> [c0, c1, c2] = [1, 0, 1]
        init_norm_alg(&mut sieve, &coeffs, 1.0, 0.0, 0.0, 1.0, 0, 10, 1.0);
        // At i=5 (index 15): |25| -> log2(25) ~ 4.64 -> 4 or 5
        assert!(
            sieve[15] >= 4 && sieve[15] <= 5,
            "norm at i=5 should be ~4.64, got {}",
            sieve[15]
        );
    }

    #[test]
    fn test_init_norm_monotonic_linear() {
        // Rational norm |i - 50| with identity lattice
        let mut sieve = vec![0u8; 100];
        init_norm_rat(&mut sieve, -50.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1, 50, 1.0);
        // Should decrease toward i=50 (index 100), increase away
        // At i=0 (index 50): |0 - 50| = 50, log2(50) ~ 5.6
        // At i=-40 (index 10): |-40 - 50| = 90, log2(90) ~ 6.5
        assert!(sieve[10] > sieve[50] || sieve[50] == 0);
    }

    #[test]
    fn test_init_norm_rat_zero_norm() {
        // F_rat = i - 5, identity lattice, j=1, g0=-5, g1=1
        let mut sieve = vec![100u8; 10]; // i in [-5, 5)
        init_norm_rat(&mut sieve, -5.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1, 5, 1.0);
        // At i=5 (index 10): out of range
        // At i=0 (index 5): |0 - 5| = 5 -> log2(5) ~ 2.32 -> 2
        assert!(sieve[5] >= 2 && sieve[5] <= 3);
        // At i=5 is just past the end. At i=4 (index 9): |4 - 5| = 1 -> log2(1) = 0 -> 0
        assert_eq!(sieve[9], 0, "norm at i=4 should be 0 (|F|=1, log2(1)=0)");
    }

    #[test]
    fn test_init_norm_alg_with_b_nonzero() {
        // f(x) = x^2 + 1, identity lattice, j=1
        // F(a,b) = a^2 + b^2, with identity lattice: a=i, b=j=1
        // F(i,1) = i^2 + 1
        let mut sieve = vec![0u8; 20]; // i in [-10, 10)
        let coeffs = [1i64, 0, 1]; // [c0, c1, c2] = [1, 0, 1]
        init_norm_alg(&mut sieve, &coeffs, 1.0, 0.0, 0.0, 1.0, 1, 10, 1.0);
        // At i=0 (index 10): |0 + 1| = 1, log2(1) = 0 -> 0
        assert_eq!(sieve[10], 0, "norm at i=0, j=1 should be 0 for x^2+1");
        // At i=3 (index 13): |9 + 1| = 10, log2(10) ~ 3.32 -> 3
        assert!(
            sieve[13] >= 3 && sieve[13] <= 4,
            "norm at i=3 should be ~3.32, got {}",
            sieve[13]
        );
    }

    #[test]
    fn test_init_norm_alg_degree_3() {
        // f(x) = x^3 - x + 1, coeffs: [1, -1, 0, 1]
        // F(a,b) = a^3 - a*b^2 + b^3
        // Identity lattice, j=0: F(i,0) = i^3 (c_d=1, d=3)
        let mut sieve = vec![0u8; 20]; // i in [-10, 10)
        let coeffs = [1i64, -1, 0, 1];
        init_norm_alg(&mut sieve, &coeffs, 1.0, 0.0, 0.0, 1.0, 0, 10, 1.0);
        // At i=4 (index 14): |64| -> log2(64) = 6.0 -> 6
        assert_eq!(sieve[14], 6, "norm at i=4 should be 6, got {}", sieve[14]);
        // At i=8 (index 18): |512| -> log2(512) = 9.0 -> 9
        assert_eq!(sieve[18], 9, "norm at i=8 should be 9, got {}", sieve[18]);
    }

    #[test]
    fn test_init_norm_rat_scale() {
        // Test that scale factor works correctly
        let mut sieve = vec![0u8; 20];
        init_norm_rat(&mut sieve, -10.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1, 10, 2.0);
        // At i=0 (index 10): |0 - 10| = 10, log2(10)*2 ~ 6.64 -> 6
        assert!(
            sieve[10] >= 6 && sieve[10] <= 7,
            "norm at i=0 with scale=2 should be ~6.64, got {}",
            sieve[10]
        );
    }

    #[test]
    fn test_init_norm_alg_empty_coeffs() {
        let mut sieve = vec![100u8; 10];
        init_norm_alg(&mut sieve, &[], 1.0, 0.0, 0.0, 1.0, 0, 5, 1.0);
        for &v in &sieve {
            assert_eq!(v, 0);
        }
    }

    #[test]
    fn test_init_norm_rat_non_identity_lattice() {
        // q-lattice: a0=2, b0=1, a1=1, b1=3, j=2
        // F_rat = g1*(2*i + 1*2) + g0*(1*i + 3*2) = (2 + g0)i + (2 + 6*g0)
        // With g1=1, g0=-1: slope = 2 + (-1)*1 = 1, intercept = (1*1 + (-1)*3)*2 = -4
        // F_rat = i - 4
        let mut sieve = vec![0u8; 20]; // i in [-10, 10)
        init_norm_rat(&mut sieve, -1.0, 1.0, 2.0, 1.0, 1.0, 3.0, 2, 10, 1.0);
        // At i=4 (index 14): |4 - 4| = 0 -> 0
        assert_eq!(sieve[14], 0, "norm at i=4 should be 0");
        // At i=0 (index 10): |0 - 4| = 4 -> log2(4) = 2 -> 2
        assert_eq!(sieve[10], 2, "norm at i=0 should be 2, got {}", sieve[10]);
    }
}
