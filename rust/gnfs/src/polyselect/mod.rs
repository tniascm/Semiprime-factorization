pub mod alpha;
pub mod dickman;
pub mod murphy_e;
pub mod rotation;
pub use alpha::murphy_alpha;

use crate::arith::nth_root;
use crate::types::PolynomialPair;
use rayon::prelude::*;
use rug::ops::Pow;
use rug::Integer;

/// Choose polynomial degree based on digit count of N.
pub fn choose_degree(digits: u32) -> u32 {
    if digits <= 50 {
        3
    } else if digits <= 100 {
        4
    } else {
        5
    }
}

/// Base-m polynomial selection.
///
/// Given N and degree d, compute m = floor(N^(1/d)), then express N in base m:
///   N = c_d * m^d + c_{d-1} * m^{d-1} + ... + c_0
/// This gives f(x) = c_d * x^d + ... + c_0 with f(m) = N ≡ 0 (mod N).
/// Using m = floor(N^{1/d}) ensures the leading coefficient c_d = 1 (monic)
/// for all N > 2^d, which is required for correct algebraic arithmetic in Z[α]/(f).
/// The rational polynomial is g(x) = x - m, so g(m) = 0.
/// We store: g0 = -m, g1 = 1.
pub fn select_base_m(n: &Integer, degree: u32) -> PolynomialPair {
    select_base_m_variant(n, degree, 0)
}

/// Base-m polynomial selection with variant index.
///
/// Variant 0: standard m = floor(N^{1/d}) (adjusted up for monic).
/// Variant k > 0: try m = standard_m - k, using a different number field.
/// This is critical when the standard polynomial produces a degenerate number
/// field where the algebraic square root always gives trivial gcd.
pub fn select_base_m_variant(n: &Integer, degree: u32, variant: u32) -> PolynomialPair {
    let d = degree as usize;
    let mut m = nth_root(n, degree);
    // Ensure monic: if leading coefficient > 1, increment m until it becomes 1
    loop {
        let m_pow_d = Integer::from(m.clone().pow(degree));
        let lead = Integer::from(n / &m_pow_d);
        if lead <= 1 {
            break;
        }
        m += 1;
    }

    // For variant > 0, decrease m to get a different polynomial
    if variant > 0 {
        m -= Integer::from(variant);
        // Verify still monic (leading coefficient = 1)
        let m_pow_d = Integer::from(m.clone().pow(degree));
        let lead = Integer::from(n / &m_pow_d);
        if lead != 1 {
            // Non-monic with this m, fall back to standard
            return select_base_m_variant(n, degree, 0);
        }
    }

    // Express N in base m
    let mut coeffs = Vec::with_capacity(d + 1);
    let mut remaining = n.clone();

    if m <= 1 {
        // Degenerate: just use N as constant coefficient
        coeffs.push(remaining);
        for _ in 1..=d {
            coeffs.push(Integer::from(0));
        }
    } else {
        loop {
            let (quot, rem) = remaining.div_rem_euc(m.clone());
            coeffs.push(rem);
            remaining = quot;
            if remaining == 0 {
                break;
            }
        }
        // Pad to exactly d+1 coefficients
        while coeffs.len() <= d {
            coeffs.push(Integer::from(0));
        }
    }

    // Center coefficients c_0 through c_{d-1} for smaller norms.
    if m > 1 {
        let half_m = Integer::from(&m / 2);
        for i in 0..d {
            if coeffs[i] > half_m {
                coeffs[i] -= &m;
                coeffs[i + 1] += 1;
            }
        }
    }

    // Verify: f(m) should equal N
    debug_assert!({
        let mut val = Integer::from(0);
        let mut m_pow = Integer::from(1);
        for c in &coeffs {
            val += c * &m_pow;
            m_pow *= &m;
        }
        val == *n
    });

    let neg_m = Integer::from(-&m);
    PolynomialPair::new(&coeffs, &neg_m, &Integer::from(1), &m, n)
}

/// Generate a polynomial pair with given leading coefficient `ad`.
///
/// Computes m = floor((N/ad)^(1/d)), then constructs f(x) = ad*x^d + ... + c_0
/// via base-m expansion such that f(m) = N.
pub fn select_polynomial_with_ad(n: &Integer, degree: u32, ad: u64) -> Option<PolynomialPair> {
    let d = degree as usize;
    let ad_int = Integer::from(ad);

    // m = floor((N / ad)^(1/d))
    let n_over_ad = Integer::from(n / &ad_int);
    let m = nth_root(&n_over_ad, degree);

    if m < 2 {
        return None;
    }

    // Try m and m-1 to find a valid base-m expansion
    for m_candidate in [m.clone(), Integer::from(&m - 1)] {
        if m_candidate < 2 {
            continue;
        }
        let m_pow_d = Integer::from(m_candidate.clone().pow(degree));
        let remaining = Integer::from(n - &ad_int * &m_pow_d);
        if remaining < 0 {
            continue;
        }

        if let Some(poly) = build_poly_from_remainder(n, &m_candidate, &ad_int, d, &remaining) {
            return Some(poly);
        }
    }

    None
}

fn build_poly_from_remainder(
    n: &Integer,
    m: &Integer,
    ad: &Integer,
    d: usize,
    remainder: &Integer,
) -> Option<PolynomialPair> {
    let mut coeffs = Vec::with_capacity(d + 1);
    let mut rem = remainder.clone();

    // Extract d coefficients c_0 through c_{d-1} via base-m expansion
    for _ in 0..d {
        let (quot, r) = rem.div_rem_euc(m.clone());
        coeffs.push(r);
        rem = quot;
    }

    // rem should be 0 if expansion is exact
    if rem != 0 {
        return None;
    }

    coeffs.push(ad.clone());

    // Center coefficients c_0 through c_{d-1}: if c_i > m/2, replace with
    // c_i - m and carry +1 to c_{i+1}. This halves coefficient magnitudes,
    // reducing norms and improving smoothness probability.
    let half_m = Integer::from(m / 2);
    for i in 0..d {
        if coeffs[i] > half_m {
            coeffs[i] -= m;
            coeffs[i + 1] += 1;
        }
    }

    // Verify: f(m) = N
    debug_assert!({
        let mut val = Integer::from(0);
        let mut m_pow = Integer::from(1);
        for c in &coeffs {
            val += c * &m_pow;
            m_pow *= m;
        }
        val == *n
    });

    let neg_m = Integer::from(-m);
    Some(PolynomialPair::new(&coeffs, &neg_m, &Integer::from(1), m, n))
}

/// Select the best polynomial for N by searching over leading coefficients
/// and applying root optimization.
///
/// Parameters match CADO's polyselect:
/// - admax: maximum leading coefficient to try
/// - incr: step size for ad sweep
/// - ropteffort: rotation search effort (multiplied by 50 for range)
/// - nrkeep: number of top polynomials to retain
pub fn select_best_polynomial(
    n: &Integer,
    degree: u32,
    admax: u64,
    incr: u64,
    ropteffort: f64,
    nrkeep: usize,
    lim: u64,
) -> Vec<PolynomialPair> {
    let alpha_bound = 200u64;
    let bf = lim as f64;
    let bg = lim as f64;
    let rotation_range = (50.0 * ropteffort) as i64;

    // Two-phase approach:
    // Phase 1: Score all candidates by (lognorm + alpha) — no rotation.
    //   Alpha (root properties) is cheap; rotation is expensive.
    // Phase 2: Full rotation + Murphy E for top candidates only.
    let phase2_keep = 100.max(nrkeep * 10);
    let mut phase1: Vec<(f64, Vec<i64>, Vec<i64>, PolynomialPair)> = Vec::new();

    // Monic variants
    for v in 0..5u32 {
        let poly = select_base_m_variant(n, degree, v);
        let f_i64 = poly_to_i64(&poly);
        let g_i64 = g_to_i64(&poly);
        if f_i64.is_empty() {
            continue;
        }
        let skew = murphy_e::optimal_skewness(&f_i64);
        let score = combined_size_score(&f_i64, skew, alpha_bound);
        phase1.push((score, f_i64, g_i64, poly));
    }

    // Non-monic: sweep ad values (parallel)
    let ad_values: Vec<u64> = (1..=admax / incr).map(|i| i * incr).collect();
    let nonmonic: Vec<_> = ad_values
        .par_iter()
        .filter_map(|&ad| {
            let poly = select_polynomial_with_ad(n, degree, ad)?;
            let f_i64 = poly_to_i64(&poly);
            let g_i64 = g_to_i64(&poly);
            if f_i64.is_empty() {
                return None;
            }
            let skew = murphy_e::optimal_skewness(&f_i64);
            let score = combined_size_score(&f_i64, skew, alpha_bound);
            Some((score, f_i64, g_i64, poly))
        })
        .collect();
    phase1.extend(nonmonic);

    let total_candidates = phase1.len();

    // Screen to phase2_keep by combined score (ascending — lower is better)
    if total_candidates > phase2_keep {
        phase1.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        phase1.truncate(phase2_keep);
    }

    // Phase 2: Full rotation + Murphy E (parallel)
    let candidates: Vec<(f64, PolynomialPair)> = phase1
        .par_iter()
        .map(|(_, f_i64, g_i64, poly)| {
            let is_monic = f_i64.last().map(|&c| c == 1).unwrap_or(false);

            let final_f = if !is_monic && rotation_range > 0 && g_i64.len() == 2 {
                let (rotated, _, _) =
                    rotation::optimize_rotation(f_i64, g_i64, rotation_range, alpha_bound);
                rotated
            } else {
                f_i64.clone()
            };

            let skew = murphy_e::optimal_skewness(&final_f);
            let e = murphy_e::murphy_e(&final_f, g_i64, skew, bf, bg, alpha_bound);

            let final_poly = if !is_monic {
                rebuild_poly_with_coeffs(poly, &final_f, n)
            } else {
                poly.clone()
            };
            (e, final_poly)
        })
        .collect();

    // Sort by Murphy E (descending — higher is better)
    let mut candidates = candidates;
    candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    candidates.truncate(nrkeep);

    if !candidates.is_empty() {
        eprintln!(
            "  polyselect: {} candidates -> {} screened -> {} kept, best E = {:.6e}",
            total_candidates,
            phase1.len(),
            candidates.len(),
            candidates.first().map(|(e, _)| *e).unwrap_or(0.0),
        );
    }

    candidates.into_iter().map(|(_, p)| p).collect()
}

/// Combined size + root score for fast Phase 1 screening.
/// Score = lognorm + alpha (lower is better).
/// lognorm measures geometric norm size; alpha measures root properties.
/// Both contribute to smoothness probability without expensive rotation.
fn combined_size_score(f_coeffs: &[i64], skewness: f64, alpha_bound: u64) -> f64 {
    let lognorm = lognorm_score(f_coeffs, skewness);
    let alpha = alpha::murphy_alpha(f_coeffs, alpha_bound);
    // Both in log scale; alpha is typically negative for good polys
    lognorm + alpha
}

/// Average log|F(x,y)| over sample points on the sieve ellipse.
fn lognorm_score(f_coeffs: &[i64], skewness: f64) -> f64 {
    let k = 64;
    let area = 1e16_f64;
    let x_scale = (area * skewness).sqrt();
    let y_scale = (area / skewness).sqrt();

    let mut sum = 0.0;
    for i in 0..k {
        let theta = std::f64::consts::PI / (k as f64) * (i as f64 + 0.5);
        let xi = x_scale * theta.cos();
        let yi = y_scale * theta.sin();

        let d = f_coeffs.len() - 1;
        let mut result = 0.0f64;
        let mut x_pow = 1.0f64;
        for (j, &c) in f_coeffs.iter().enumerate() {
            let y_pow = yi.powi((d - j) as i32);
            result += c as f64 * x_pow * y_pow;
            x_pow *= xi;
        }
        sum += result.abs().max(1.0).ln();
    }
    sum / k as f64
}

fn poly_to_i64(poly: &PolynomialPair) -> Vec<i64> {
    poly.f_coeffs_str
        .iter()
        .map(|s| s.parse::<i64>().unwrap_or(0))
        .collect()
}

fn g_to_i64(poly: &PolynomialPair) -> Vec<i64> {
    poly.g_coeffs_str
        .iter()
        .map(|s| s.parse::<i64>().unwrap_or(0))
        .collect()
}

#[allow(dead_code)]
/// Translate polynomial: compute f(x + k) using Horner's shift algorithm.
///
/// Coefficients stored low-degree first: [c0, c1, ..., cd].
/// The translation preserves f(m) = N with new m' = m - k.
fn translate_polynomial(f: &[i64], k: i64) -> Vec<i64> {
    if k == 0 {
        return f.to_vec();
    }
    let d = f.len() - 1;
    let mut result = f.to_vec();
    // Taylor shift algorithm: compute f(x+k) in-place.
    // Outer loop from d-1 down to 0, inner loop from j to d-1.
    for j in (0..d).rev() {
        for i in j..d {
            let product = (result[i + 1] as i128) * (k as i128);
            let sum = (result[i] as i128) + product;
            if sum > i64::MAX as i128 || sum < i64::MIN as i128 {
                return f.to_vec(); // overflow — return untranslated
            }
            result[i] = sum as i64;
        }
    }
    result
}

#[allow(dead_code)]
/// Translate linear polynomial g(x) = g0 + g1*x by k: g(x+k) = (g0 + g1*k) + g1*x.
fn translate_g(g: &[i64], k: i64) -> Vec<i64> {
    if g.len() != 2 || k == 0 {
        return g.to_vec();
    }
    vec![g[0] + g[1] * k, g[1]]
}

#[allow(dead_code)]
/// L2 norm of polynomial at given skewness: sum c_i^2 * s^{2(i - d/2)}.
/// Lower is better (smaller norms → higher smoothness probability).
fn l2_skew_norm(f: &[i64], skewness: f64) -> f64 {
    let d = f.len() - 1;
    let half_d = d as f64 / 2.0;
    let mut sum = 0.0;
    for (i, &c) in f.iter().enumerate() {
        let s_pow = skewness.powf(i as f64 - half_d);
        sum += (c as f64 * s_pow) * (c as f64 * s_pow);
    }
    sum
}

#[allow(dead_code)]
/// Find the best translation k for a polynomial to balance norms.
///
/// For degree d, the ideal translation zeroes c_{d-1}: k_opt = -c_{d-1} / (d * c_d).
/// We search around that point using the cheap L2 skew-norm as objective.
fn optimize_translation(
    f: &[i64],
    g: &[i64],
    _bf: f64,
    _bg: f64,
    _alpha_bound: u64,
) -> (Vec<i64>, Vec<i64>, i64) {
    let d = f.len() - 1;
    if d < 2 || f[d] == 0 {
        return (f.to_vec(), g.to_vec(), 0);
    }

    // Optimal k to zero c_{d-1}
    let k_center_f = -(f[d - 1] as f64) / (d as f64 * f[d] as f64);
    let k_center = k_center_f.round() as i64;

    // Narrow search: ±20 around the analytically optimal point.
    let search_half = 20i64;

    let skew_orig = murphy_e::optimal_skewness(f);
    let mut best_score = l2_skew_norm(f, skew_orig);
    let mut best_f = f.to_vec();
    let mut best_g = g.to_vec();
    let mut best_k = 0i64;

    for k in (k_center - search_half)..=(k_center + search_half) {
        if k == 0 {
            continue;
        }
        let f_t = translate_polynomial(f, k);
        // Overflow check
        if f_t == f.to_vec() {
            continue;
        }
        let g_t = translate_g(g, k);
        let skew = murphy_e::optimal_skewness(&f_t);
        let score = l2_skew_norm(&f_t, skew);
        if score < best_score {
            best_score = score;
            best_f = f_t;
            best_g = g_t;
            best_k = k;
        }
    }

    (best_f, best_g, best_k)
}

fn rebuild_poly_with_coeffs(
    original: &PolynomialPair,
    new_f_coeffs: &[i64],
    n: &Integer,
) -> PolynomialPair {
    let coeffs: Vec<Integer> = new_f_coeffs.iter().map(|&c| Integer::from(c)).collect();
    let g0: Integer = original.g_coeffs_str[0].parse().unwrap();
    let g1: Integer = original.g_coeffs_str[1].parse().unwrap();
    let m: Integer = original.m_str.parse().unwrap();
    PolynomialPair::new(&coeffs, &g0, &g1, &m, n)
}

#[allow(dead_code)]
/// Build a PolynomialPair from explicit f and g coefficient arrays.
/// g is linear [g0, g1], m = -g0/g1 (when g1=1, m = -g0).
fn rebuild_poly_with_g(
    f_coeffs: &[i64],
    g_coeffs: &[i64],
    n: &Integer,
) -> PolynomialPair {
    let coeffs: Vec<Integer> = f_coeffs.iter().map(|&c| Integer::from(c)).collect();
    let g0 = Integer::from(g_coeffs[0]);
    let g1 = if g_coeffs.len() > 1 {
        Integer::from(g_coeffs[1])
    } else {
        Integer::from(1)
    };
    // m = -g0 / g1 (for g1=1, m = -g0)
    let m = if g1 == 1 {
        Integer::from(-g_coeffs[0])
    } else {
        Integer::from(-g_coeffs[0]) / &g1
    };
    PolynomialPair::new(&coeffs, &g0, &g1, &m, n)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rug::Integer;

    #[test]
    fn test_base_m_selection_small() {
        // 8051 = 83 * 97
        let n = Integer::from(8051);
        let poly = select_base_m(&n, 3);
        // Verify f(m) ≡ 0 (mod N)
        let f = poly.f_coeffs();
        let m = poly.m();
        let mut val = Integer::from(0);
        let mut m_pow = Integer::from(1);
        for c in &f {
            val += c * &m_pow;
            m_pow *= &m;
        }
        assert_eq!(val % &n, 0, "f(m) must be 0 mod N");
    }

    #[test]
    fn test_base_m_selection_g_linear() {
        let n = Integer::from(15347);
        let poly = select_base_m(&n, 3);
        // g(m) = g0 + g1 * m should be divisible by N
        let g0 = poly.g0();
        let g1 = poly.g1();
        let m = poly.m();
        let gm = Integer::from(&g0 + &g1 * &m);
        assert_eq!(gm % &n, 0, "g(m) must be 0 mod N");
    }

    #[test]
    fn test_choose_degree() {
        assert_eq!(choose_degree(50), 3);
        assert_eq!(choose_degree(100), 4);
        assert_eq!(choose_degree(150), 5);
        assert_eq!(choose_degree(200), 5);
        assert_eq!(choose_degree(300), 5);
    }

    #[test]
    fn test_base_m_60_digit() {
        // A 60-digit semiprime
        let n: Integer = "523022617466601111760007224100074291200000259"
            .parse()
            .unwrap();
        let d = choose_degree(n.significant_bits() as u32 * 3 / 10);
        let poly = select_base_m(&n, d);
        let f = poly.f_coeffs();
        let m = poly.m();
        let mut val = Integer::from(0);
        let mut m_pow = Integer::from(1);
        for c in &f {
            val += c * &m_pow;
            m_pow *= &m;
        }
        assert_eq!(val % &n, 0);
    }

    #[test]
    fn test_select_polynomial_with_ad_monic() {
        let n = Integer::from_str_radix("684217602914977371691118975023", 10).unwrap();
        // ad=1 should produce a valid polynomial
        let poly = select_polynomial_with_ad(&n, 3, 1).unwrap();
        assert_eq!(poly.degree, 3);
        // Verify f(m) = N
        let m = poly.m();
        let coeffs = poly.f_coeffs();
        let mut val = Integer::from(0);
        let mut m_pow = Integer::from(1);
        for c in &coeffs {
            val += c * &m_pow;
            m_pow *= &m;
        }
        assert_eq!(val, n, "f(m) must equal N");
    }

    #[test]
    fn test_nonmonic_polynomial_ad_60() {
        let n = Integer::from_str_radix("684217602914977371691118975023", 10).unwrap();
        let poly = select_polynomial_with_ad(&n, 3, 60).unwrap();
        assert_eq!(poly.degree, 3);
        // Leading coefficient should be 60
        let lead = poly.f_coeffs().last().unwrap().clone();
        assert_eq!(lead, Integer::from(60));
        // Verify f(m) = N
        let m = poly.m();
        let coeffs = poly.f_coeffs();
        let mut val = Integer::from(0);
        let mut m_pow = Integer::from(1);
        for c in &coeffs {
            val += c * &m_pow;
            m_pow *= &m;
        }
        assert_eq!(val, n, "f(m) must equal N for ad=60");
    }

    #[test]
    fn test_nonmonic_sweep_produces_diverse_polys() {
        let n = Integer::from_str_radix("684217602914977371691118975023", 10).unwrap();
        let mut valid = 0;
        for ad in (20..=200).step_by(20) {
            if select_polynomial_with_ad(&n, 3, ad).is_some() {
                valid += 1;
            }
        }
        assert!(
            valid >= 5,
            "At least half of ad values should produce valid polys, got {}",
            valid
        );
    }

    #[test]
    fn test_select_best_polynomial() {
        let n = Integer::from_str_radix("684217602914977371691118975023", 10).unwrap();
        // Use small parameters for fast debug-mode testing:
        // admax=100 (5 ad values), ropteffort=0 (skip rotation), nrkeep=3
        let polys = select_best_polynomial(&n, 3, 100, 20, 0.0, 3, 30_000);
        assert!(!polys.is_empty(), "Should find at least one polynomial");
        // Verify best polynomial f(m) = N
        let best = &polys[0];
        let m = best.m();
        let coeffs = best.f_coeffs();
        let mut val = Integer::from(0);
        let mut m_pow = Integer::from(1);
        for c in &coeffs {
            val += c * &m_pow;
            m_pow *= &m;
        }
        assert_eq!(val, n, "Best polynomial must satisfy f(m) = N");
    }

    #[test]
    fn test_translate_polynomial_identity() {
        let f = vec![5i64, 3, 1]; // x^2 + 3x + 5
        let f_t = translate_polynomial(&f, 0);
        assert_eq!(f_t, f);
    }

    #[test]
    fn test_translate_polynomial_quadratic() {
        // f(x) = x^2 + 3x + 5 = [5, 3, 1], k=2
        // f(x+2) = (x+2)^2 + 3(x+2) + 5 = x^2 + 7x + 15
        let f = vec![5i64, 3, 1];
        let f_t = translate_polynomial(&f, 2);
        assert_eq!(f_t, vec![15, 7, 1]);
    }

    #[test]
    fn test_translate_polynomial_cubic() {
        // f(x) = x^3 = [0, 0, 0, 1], k=1
        // f(x+1) = (x+1)^3 = x^3 + 3x^2 + 3x + 1
        let f = vec![0i64, 0, 0, 1];
        let f_t = translate_polynomial(&f, 1);
        assert_eq!(f_t, vec![1, 3, 3, 1]);
    }

    #[test]
    fn test_translate_polynomial_preserves_evaluation() {
        // f(x) = 2x^3 - 5x^2 + 3x + 7
        let f = vec![7i64, 3, -5, 2];
        let k = 3i64;
        let f_t = translate_polynomial(&f, k);
        // f(10) should equal f_t(10 - k) = f_t(7)
        let eval_f_10: i64 = f.iter().enumerate().map(|(i, &c)| c * 10i64.pow(i as u32)).sum();
        let eval_ft_7: i64 = f_t.iter().enumerate().map(|(i, &c)| c * 7i64.pow(i as u32)).sum();
        assert_eq!(eval_f_10, eval_ft_7, "f(10) must equal f_translated(7)");
    }

    #[test]
    fn test_translate_g() {
        let g = vec![-100i64, 1]; // g(x) = x - 100
        let g_t = translate_g(&g, 5);
        // g(x+5) = (x+5) - 100 = x - 95
        assert_eq!(g_t, vec![-95, 1]);
    }

    #[test]
    fn test_optimize_translation_improves_or_matches() {
        let n = Integer::from_str_radix("684217602914977371691118975023", 10).unwrap();
        let poly = select_polynomial_with_ad(&n, 3, 900).unwrap();
        let f = poly_to_i64(&poly);
        let g = g_to_i64(&poly);
        let bf = 30000.0;
        let bg = 30000.0;

        let skew_before = murphy_e::optimal_skewness(&f);
        let e_before = murphy_e::murphy_e(&f, &g, skew_before, bf, bg, 200);

        let (f_opt, g_opt, _k) = optimize_translation(&f, &g, bf, bg, 200);
        let skew_after = murphy_e::optimal_skewness(&f_opt);
        let e_after = murphy_e::murphy_e(&f_opt, &g_opt, skew_after, bf, bg, 200);

        assert!(
            e_after >= e_before * 0.99,
            "Translation should not significantly worsen E: {:.6e} -> {:.6e}",
            e_before,
            e_after
        );
    }

    #[test]
    fn test_translated_polynomial_satisfies_fm_eq_n() {
        let n = Integer::from_str_radix("684217602914977371691118975023", 10).unwrap();
        let poly = select_polynomial_with_ad(&n, 3, 900).unwrap();
        let f = poly_to_i64(&poly);
        let g = g_to_i64(&poly);

        let (f_opt, g_opt, _k) = optimize_translation(&f, &g, 30000.0, 30000.0, 200);

        // Verify: the translated polynomial still satisfies Res(f,g) ≡ 0 mod N.
        // For g(x) = g0 + g1*x with g1=1: f(-g0) should ≡ 0 (mod N).
        let m_new = -g_opt[0]; // g1=1, so m = -g0
        let mut val = Integer::from(0);
        let mut m_pow = Integer::from(1);
        let m_int = Integer::from(m_new);
        for &c in &f_opt {
            val += Integer::from(c) * &m_pow;
            m_pow *= &m_int;
        }
        assert_eq!(
            Integer::from(&val % &n),
            0,
            "Translated f(m') must be 0 mod N"
        );
    }
}
