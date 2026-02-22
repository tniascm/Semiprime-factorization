//! Complex number arithmetic using (f64, f64) tuples.
//!
//! We use a simple representation: Complex = (re, im) to avoid external dependencies.

/// A complex number represented as (real, imaginary).
pub type Complex = (f64, f64);

/// The complex number zero.
pub const ZERO: Complex = (0.0, 0.0);

/// The complex number one.
pub const ONE: Complex = (1.0, 0.0);

/// The imaginary unit i.
pub const I: Complex = (0.0, 1.0);

/// Add two complex numbers.
#[inline]
pub fn cadd(a: Complex, b: Complex) -> Complex {
    (a.0 + b.0, a.1 + b.1)
}

/// Subtract two complex numbers.
#[inline]
pub fn csub(a: Complex, b: Complex) -> Complex {
    (a.0 - b.0, a.1 - b.1)
}

/// Multiply two complex numbers.
#[inline]
pub fn cmul(a: Complex, b: Complex) -> Complex {
    (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
}

/// Divide two complex numbers.
#[inline]
pub fn cdiv(a: Complex, b: Complex) -> Complex {
    let denom = b.0 * b.0 + b.1 * b.1;
    (
        (a.0 * b.0 + a.1 * b.1) / denom,
        (a.1 * b.0 - a.0 * b.1) / denom,
    )
}

/// Complex conjugate.
#[inline]
pub fn conj(a: Complex) -> Complex {
    (a.0, -a.1)
}

/// Squared magnitude |z|^2 = re^2 + im^2.
#[inline]
pub fn cnorm_sq(a: Complex) -> f64 {
    a.0 * a.0 + a.1 * a.1
}

/// Magnitude |z|.
#[inline]
pub fn cabs(a: Complex) -> f64 {
    cnorm_sq(a).sqrt()
}

/// Complex exponential e^z = e^re * (cos(im) + i*sin(im)).
#[inline]
pub fn cexp(z: Complex) -> Complex {
    let r = z.0.exp();
    (r * z.1.cos(), r * z.1.sin())
}

/// Complex exponential e^{i*theta}.
#[inline]
pub fn cis(theta: f64) -> Complex {
    (theta.cos(), theta.sin())
}

/// Multiply a complex number by a real scalar.
#[inline]
pub fn cscale(s: f64, z: Complex) -> Complex {
    (s * z.0, s * z.1)
}

/// Raise a complex number to a real power: z^s for real s.
/// Uses z^s = |z|^s * e^{i*s*arg(z)}.
pub fn cpow_real(z: Complex, s: f64) -> Complex {
    let r = cabs(z);
    if r < 1e-15 {
        return ZERO;
    }
    let theta = z.1.atan2(z.0);
    let new_r = r.powf(s);
    (new_r * (s * theta).cos(), new_r * (s * theta).sin())
}

/// Sum of a slice of complex numbers.
pub fn csum(values: &[Complex]) -> Complex {
    values.iter().fold(ZERO, |acc, &v| cadd(acc, v))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_basic_arithmetic() {
        let a = (3.0, 4.0);
        let b = (1.0, 2.0);

        let sum = cadd(a, b);
        assert!((sum.0 - 4.0).abs() < 1e-10);
        assert!((sum.1 - 6.0).abs() < 1e-10);

        let prod = cmul(a, b);
        // (3+4i)(1+2i) = 3+6i+4i+8i^2 = -5+10i
        assert!((prod.0 - (-5.0)).abs() < 1e-10);
        assert!((prod.1 - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_cis() {
        let z = cis(PI / 2.0);
        assert!(z.0.abs() < 1e-10); // cos(pi/2) = 0
        assert!((z.1 - 1.0).abs() < 1e-10); // sin(pi/2) = 1
    }

    #[test]
    fn test_norm() {
        let z = (3.0, 4.0);
        assert!((cnorm_sq(z) - 25.0).abs() < 1e-10);
        assert!((cabs(z) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_exp() {
        // e^{i*pi} = -1
        let z = (0.0, PI);
        let result = cexp(z);
        assert!((result.0 - (-1.0)).abs() < 1e-10);
        assert!(result.1.abs() < 1e-10);
    }
}
