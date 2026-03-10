use crate::polyselect::alpha::murphy_alpha;
use crate::polyselect::murphy_e;

/// Apply rotation: f'(x) = f(x) + (u*x + v) * g(x)
///
/// This preserves f(m) = N since g(m) = 0.
/// g is assumed to be linear: g(x) = g[0] + g[1]*x.
/// Coefficients stored low-degree first: [c0, c1, ..., cd].
pub fn apply_rotation(f: &[i64], g: &[i64], u: i64, v: i64) -> Vec<i64> {
    assert!(g.len() == 2, "g must be linear for rotation");
    let mut result = f.to_vec();

    // Add v*g: result[i] += v * g[i]
    if v != 0 {
        for (i, &gi) in g.iter().enumerate() {
            if i < result.len() {
                result[i] += v * gi;
            }
        }
    }

    // Add u*x*g: result[i+1] += u * g[i]
    if u != 0 {
        while result.len() < g.len() + 1 {
            result.push(0);
        }
        for (i, &gi) in g.iter().enumerate() {
            result[i + 1] += u * gi;
        }
    }

    // Trim trailing zeros (but keep at least degree+1 coefficients)
    while result.len() > 1 && *result.last().unwrap() == 0 {
        result.pop();
    }

    result
}

/// Fast combined score for rotation search: lognorm + lightweight alpha.
///
/// Uses `lognorm` (O(d×k) per evaluation) + `murphy_alpha` with a small
/// prime bound (only primes up to 29 = 10 primes vs 45 for bound=200).
/// This captures root properties at ~5x lower cost than full alpha.
fn rotation_score(f: &[i64]) -> f64 {
    let skew = murphy_e::optimal_skewness(f);
    // Average log|F(x,y)| over ellipse samples — lower is better.
    let k = 32;
    let area = 1e16_f64;
    let x_scale = (area * skew).sqrt();
    let y_scale = (area / skew).sqrt();
    let d = f.len() - 1;
    let mut sum = 0.0;
    for i in 0..k {
        let theta = std::f64::consts::PI / (k as f64) * (i as f64 + 0.5);
        let xi = x_scale * theta.cos();
        let yi = y_scale * theta.sin();
        let mut val = 0.0f64;
        let mut x_pow = 1.0f64;
        for (j, &c) in f.iter().enumerate() {
            let y_pow = yi.powi((d - j) as i32);
            val += c as f64 * x_pow * y_pow;
            x_pow *= xi;
        }
        sum += val.abs().max(1.0).ln();
    }
    let lognorm = sum / k as f64;
    // Lightweight alpha: only primes up to 29 (10 primes)
    let alpha = murphy_alpha(f, 29);
    lognorm + alpha
}

/// Search for the best rotation (u, v) minimizing lognorm + light alpha.
///
/// Uses fast combined scoring for the grid search (lognorm + alpha with
/// primes up to 29). This is ~25x faster than full alpha (bound=200)
/// while still capturing root properties from the most impactful primes.
/// Returns (best_f, best_u, best_v).
pub fn optimize_rotation(
    f: &[i64],
    g: &[i64],
    search_range: i64,
    _alpha_bound: u64,
) -> (Vec<i64>, i64, i64) {
    let mut best_f = f.to_vec();
    let mut best_score = rotation_score(f);
    let mut best_u = 0i64;
    let mut best_v = 0i64;

    let v_range = search_range;
    let u_range = search_range / 10;

    for v in -v_range..=v_range {
        let f_v = apply_rotation(f, g, 0, v);
        let score_v = rotation_score(&f_v);

        if score_v < best_score {
            best_score = score_v;
            best_f = f_v.clone();
            best_u = 0;
            best_v = v;
        }

        if u_range > 0 {
            for u in -u_range..=u_range {
                if u == 0 {
                    continue;
                }
                let f_uv = apply_rotation(f, g, u, v);
                let score_uv = rotation_score(&f_uv);
                if score_uv < best_score {
                    best_score = score_uv;
                    best_f = f_uv;
                    best_u = u;
                    best_v = v;
                }
            }
        }
    }

    (best_f, best_u, best_v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rotation_preserves_identity() {
        // f(x) = x^3 + 3x^2 + 5x + 7, g(x) = x - 10, m = 10
        // f + 5*g = (7 + 5*(-10)), (5 + 5*1), 3, 1 = [-43, 10, 3, 1]
        let f = vec![7i64, 5, 3, 1];
        let g = vec![-10i64, 1];
        let f_rot = apply_rotation(&f, &g, 0, 5);
        assert_eq!(f_rot, vec![-43, 10, 3, 1]);
    }

    #[test]
    fn test_rotation_with_ux() {
        // f + 3*x*g where g = [-10, 1]
        // = f[0], f[1]+3*g[0], f[2]+3*g[1], f[3]
        let f = vec![7i64, 5, 3, 1];
        let g = vec![-10i64, 1];
        let f_rot = apply_rotation(&f, &g, 3, 0);
        assert_eq!(f_rot, vec![7, 5 + 3 * (-10), 3 + 3 * 1, 1]);
    }

    #[test]
    fn test_rotation_combined_uv() {
        // f + (2x + 3)*g where g = [-10, 1]
        // v*g: [7+3*(-10), 5+3*1, 3, 1] = [-23, 8, 3, 1]
        // u*x*g: [-23, 8+2*(-10), 3+2*1, 1] = [-23, -12, 5, 1]
        let f = vec![7i64, 5, 3, 1];
        let g = vec![-10i64, 1];
        let f_rot = apply_rotation(&f, &g, 2, 3);
        assert_eq!(f_rot, vec![-23, -12, 5, 1]);
    }

    #[test]
    fn test_optimize_rotation_does_not_worsen() {
        let f = vec![7i64, 5, 3, 1];
        let g = vec![-10i64, 1];
        let alpha_before = crate::polyselect::alpha::murphy_alpha(&f, 200);
        let (best_f, _u, _v) = optimize_rotation(&f, &g, 50, 200);
        let alpha_after = crate::polyselect::alpha::murphy_alpha(&best_f, 200);
        assert!(
            alpha_after <= alpha_before + 0.01,
            "Rotation should not worsen alpha: {} -> {}",
            alpha_before,
            alpha_after
        );
    }

    #[test]
    fn test_zero_rotation_returns_original() {
        let f = vec![7i64, 5, 3, 1];
        let g = vec![-10i64, 1];
        let f_rot = apply_rotation(&f, &g, 0, 0);
        assert_eq!(f_rot, f);
    }
}
