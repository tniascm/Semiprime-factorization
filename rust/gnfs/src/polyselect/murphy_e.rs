use crate::polyselect::alpha::murphy_alpha;
use crate::polyselect::dickman::dickman_rho;
use std::f64::consts::PI;

/// Compute Murphy's E-value for a polynomial pair (f, g).
///
/// E = (1/K) * sum_{i=0}^{K-1} rho(log|F(x_i,y_i)| / log(Bf)) * rho(log|G(x_i,y_i)| / log(Bg))
///
/// where (x_i, y_i) are sample points on an ellipse scaled by skewness,
/// and Bf/Bg are the factor bases bounds.
pub fn murphy_e(
    f_coeffs: &[i64],
    g_coeffs: &[i64],
    skewness: f64,
    bf: f64,
    bg: f64,
    alpha_bound: u64,
) -> f64 {
    let k = 1000;
    let area = 1e16_f64;

    let x_scale = (area * skewness).sqrt();
    let y_scale = (area / skewness).sqrt();

    let alpha_f = murphy_alpha(f_coeffs, alpha_bound);
    let alpha_g = murphy_alpha(g_coeffs, alpha_bound);

    let log_bf = bf.ln();
    let log_bg = bg.ln();

    let mut e_sum = 0.0;

    for i in 0..k {
        let theta = PI / (k as f64) * (i as f64 + 0.5);
        let xi = x_scale * theta.cos();
        let yi = y_scale * theta.sin();

        let log_nf = eval_homogeneous_log(f_coeffs, xi, yi);
        let log_ng = eval_homogeneous_log(g_coeffs, xi, yi);

        // Adjust by alpha (accounts for root properties)
        let uf = (log_nf + alpha_f) / log_bf;
        let ug = (log_ng + alpha_g) / log_bg;

        if uf > 0.0 && ug > 0.0 {
            e_sum += dickman_rho(uf) * dickman_rho(ug);
        }
    }

    e_sum / k as f64
}

/// Evaluate log|F(x, y)| where F is the homogenization of f.
/// F(x,y) = c_0*y^d + c_1*x*y^{d-1} + ... + c_d*x^d
fn eval_homogeneous_log(coeffs: &[i64], x: f64, y: f64) -> f64 {
    let d = coeffs.len() - 1;
    let mut result = 0.0f64;
    let mut x_pow = 1.0f64;
    for (i, &c) in coeffs.iter().enumerate() {
        let y_pow = y.powi((d - i) as i32);
        result += c as f64 * x_pow * y_pow;
        x_pow *= x;
    }
    result.abs().max(1.0).ln()
}

/// Compute optimal skewness for a polynomial.
/// For degree d: skewness ~ (|c_0| / |c_d|)^(1/d)
pub fn optimal_skewness(f_coeffs: &[i64]) -> f64 {
    let d = f_coeffs.len() - 1;
    if d == 0 {
        return 1.0;
    }
    let c0 = (f_coeffs[0] as f64).abs().max(1.0);
    let cd = (f_coeffs[d] as f64).abs().max(1.0);
    (c0 / cd).powf(1.0 / d as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_murphy_e_positive() {
        let f = vec![7i64, 5, 3, 1];
        let g = vec![-10i64, 1];
        let skew = optimal_skewness(&f);
        let e = murphy_e(&f, &g, skew, 1e5, 1e5, 200);
        assert!(e > 0.0, "Murphy E should be positive: {}", e);
    }

    #[test]
    fn test_murphy_e_better_poly_scores_higher() {
        let f_good = vec![0i64, 2, -3, 1]; // x(x-1)(x-2): 3 roots
        let f_bad = vec![1i64, 1, 0, 1]; // x^3 + x + 1: fewer roots
        let g = vec![-10i64, 1];
        let e_good = murphy_e(&f_good, &g, 1.0, 1e5, 1e5, 200);
        let e_bad = murphy_e(&f_bad, &g, 1.0, 1e5, 1e5, 200);
        assert!(
            e_good > e_bad,
            "Poly with better roots should have higher E: {} vs {}",
            e_good,
            e_bad
        );
    }

    #[test]
    fn test_optimal_skewness_symmetric() {
        // f = x^3 + 1: |c0|=|c3|=1 => skewness=1
        let s = optimal_skewness(&[1, 0, 0, 1]);
        assert!(
            (s - 1.0).abs() < 0.01,
            "Symmetric poly should have skewness~1: {}",
            s
        );
    }

    #[test]
    fn test_optimal_skewness_asymmetric() {
        // f = 1000*x^3 + 1: skew = (1/1000)^(1/3) ~ 0.1
        let s = optimal_skewness(&[1, 0, 0, 1000]);
        assert!(s < 0.5, "Asymmetric poly should have low skewness: {}", s);
    }
}
