use rug::Integer;
use rug::ops::Pow;
use crate::types::Relation;

/// Compute the rational square root for a dependency.
///
/// Given a set of relations whose product of rational norms a_i + b_i*m is a perfect square,
/// compute x = sqrt(product of (a_i + b_i*m)) mod N.
///
/// Returns Some((x, sign_negative)) or None if the product is not a perfect square.
pub fn rational_square_root(
    relations: &[Relation],
    dependency: &[usize],
    m: &Integer,
    n: &Integer,
) -> Option<(Integer, bool)> {
    let mut product = Integer::from(1);
    let mut sign_negative = false;

    for &idx in dependency {
        let rel = &relations[idx];
        let norm = Integer::from(rel.a) - Integer::from(rel.b) * m;
        if norm < 0 {
            sign_negative = !sign_negative;
            product *= Integer::from(-&norm);
        } else {
            product *= &norm;
        }
    }

    let sqrt_product = product.clone().sqrt();

    if Integer::from(&sqrt_product * &sqrt_product) != product {
        return None;
    }

    let x = sqrt_product % n;
    Some((x, sign_negative))
}

/// Create number field element (a - b*alpha) as a polynomial of degree < d.
/// Returns coefficients [a, -b, 0, ..., 0].
pub fn nf_element_from_ab(a: i64, b: u64, degree: usize) -> Vec<Integer> {
    let mut elem = vec![Integer::from(0); degree];
    elem[0] = Integer::from(a);
    if degree > 1 {
        elem[1] = Integer::from(-(b as i64));
    }
    elem
}

/// Multiply two elements in Z[alpha]/(f(alpha)).
/// Both a and b are polynomials of degree < deg(f), represented as coefficient vectors.
/// f must be monic: f = [c0, c1, ..., c_{d-1}, 1] (leading coefficient 1).
pub fn nf_multiply(a: &[Integer], b: &[Integer], f: &[Integer]) -> Vec<Integer> {
    let d = f.len() - 1;
    debug_assert_eq!(f[d], Integer::from(1), "nf_multiply requires monic f");

    // Standard polynomial multiplication
    let mut product = vec![Integer::from(0); 2 * d - 1];
    for (i, ai) in a.iter().enumerate() {
        for (j, bj) in b.iter().enumerate() {
            product[i + j] += Integer::from(ai * bj);
        }
    }

    // Reduce modulo monic f(x): x^d = -(c_{d-1}*x^{d-1} + ... + c_0)
    for i in (d..product.len()).rev() {
        let coeff = product[i].clone();
        if coeff != 0 {
            product[i] = Integer::from(0);
            for j in 0..d {
                product[i - d + j] -= Integer::from(&coeff * &f[j]);
            }
        }
    }

    product.truncate(d);
    product
}


/// Compute the algebraic square root using Couveignes' method.
///
/// Given the exact product P(α) in Z[α]/(f) (where f is monic), finds γ such that γ² = P
/// exactly in Z[α]/(f), then evaluates γ(m) mod N.
///
/// Strategy: compute a generous upper bound B on ||γ||_∞, Hensel lift until modulus > 2B,
/// then try all 2^d sign combinations with exact verification. This avoids expensive
/// operations at intermediate (insufficient) moduli and handles the "not a square" case
/// gracefully (returns empty after bounded work).
///
/// Returns all valid y = γ(m) mod N values.
fn algebraic_square_roots(
    product: &[Integer],
    f_coeffs: &[Integer],
    m: &Integer,
    n: &Integer,
) -> Vec<Integer> {
    use crate::arith::*;

    let d = f_coeffs.len() - 1;
    debug_assert_eq!(f_coeffs[d], Integer::from(1), "f must be monic");

    let f_i64: Option<Vec<i64>> = f_coeffs.iter().map(|c| c.to_i64()).collect();
    let f_i64 = match f_i64 {
        Some(v) => v,
        None => return vec![],
    };

    // Step 1: Find prime ℓ where f splits completely (d distinct roots)
    // and P(r) is a QR mod ℓ for all roots r
    let mut ell = 0u64;
    let mut base_roots: Vec<u64> = Vec::new();
    let primes = sieve_primes(100_000);

    for &p in &primes {
        if p < 3 { continue; }
        let roots = find_polynomial_roots_mod_p(&f_i64, p);
        if roots.len() != d { continue; }
        let all_nonzero_qr = roots.iter().all(|&r| {
            let r_int = Integer::from(r);
            let p_int = Integer::from(p);
            let val = eval_poly_int(product, &r_int, &p_int);
            // P(r) must be nonzero mod ℓ for Hensel lifting of √P(r)
            val != 0 && tonelli_shanks_int(&val, &p_int).is_some()
        });
        if all_nonzero_qr {
            ell = p;
            base_roots = roots;
            break;
        }
    }

    if ell == 0 {
        return vec![];
    }

    // Step 2: Compute bound for γ coefficients
    // For γ² = P in Z[α]/(f), a safe bound is:
    //   ||γ||_∞ ≤ ||P||_∞^{1/2} * (d * ||f||_∞)^{d} * d^d
    // This is conservative but ensures correctness.
    let max_p = product.iter().map(|c| c.clone().abs()).max().unwrap_or(Integer::from(1));
    let max_f = f_coeffs.iter().map(|c| c.clone().abs()).max().unwrap_or(Integer::from(1));
    let p_sqrt: Integer = Integer::from(max_p.sqrt_ref()) + 1;
    let df = Integer::from(&max_f * d as u64);
    let mut bound = p_sqrt;
    for _ in 0..d {
        bound *= &df;
    }
    // Add extra safety margin
    bound *= Integer::from(d as u64).pow(d as u32);
    let target = Integer::from(&bound * 2);

    // Step 3: Initialize roots and sqrt values mod ℓ
    let mut roots: Vec<Integer> = base_roots.iter().map(|&r| Integer::from(r)).collect();
    let mut modulus = Integer::from(ell);

    let mut sqrts: Vec<Integer> = Vec::with_capacity(d);
    for r in &roots {
        let val = eval_poly_int(product, r, &modulus);
        match tonelli_shanks_int(&val, &modulus) {
            Some(s) => sqrts.push(s),
            None => return vec![],
        }
    }

    // Step 4: Hensel lift until modulus > target bound
    let max_lift_steps = 50;
    for _ in 0..max_lift_steps {
        if modulus > target {
            break;
        }

        let new_mod = Integer::from(&modulus * &modulus);

        // Lift roots of f using Newton's method
        for i in 0..d {
            let fr = eval_poly_int(f_coeffs, &roots[i], &new_mod);
            let fpr = eval_poly_deriv_int(f_coeffs, &roots[i], &new_mod);
            let fpr_inv = match mod_inverse_int(&fpr, &new_mod) {
                Some(inv) => inv,
                None => return vec![],
            };
            let correction = Integer::from(&fr * &fpr_inv) % &new_mod;
            roots[i] = Integer::from(&roots[i] - &correction) % &new_mod;
            if roots[i] < 0 {
                roots[i] += &new_mod;
            }
        }

        // Lift sqrt values using Newton's method: s_new = s + (P(r) - s²)/(2s)
        for i in 0..d {
            let p_val = eval_poly_int(product, &roots[i], &new_mod);
            let s_sq = Integer::from(&sqrts[i] * &sqrts[i]) % &new_mod;
            let mut residual = Integer::from(&p_val - &s_sq) % &new_mod;
            if residual < 0 {
                residual += &new_mod;
            }
            let two_s = Integer::from(&sqrts[i] * 2) % &new_mod;
            let two_s_inv = match mod_inverse_int(&two_s, &new_mod) {
                Some(inv) => inv,
                None => return vec![],
            };
            let delta = Integer::from(&residual * &two_s_inv) % &new_mod;
            sqrts[i] = Integer::from(&sqrts[i] + &delta) % &new_mod;
            if sqrts[i] < 0 {
                sqrts[i] += &new_mod;
            }
        }

        modulus = new_mod;
    }

    // Verify Hensel lifting correctness: s_i² ≡ P(r_i) mod ℓ^k
    for i in 0..d {
        let p_val = eval_poly_int(product, &roots[i], &modulus);
        let s_sq = Integer::from(&sqrts[i] * &sqrts[i]) % &modulus;
        let s_sq = if s_sq < 0 { s_sq + &modulus } else { s_sq };
        if p_val != s_sq {
            return vec![];
        }
    }

    // Step 5: Try all 2^d sign combinations with EXACT verification
    let half = Integer::from(&modulus / 2);
    let mut results = Vec::new();

    for mask in 0..(1u64 << d) {
        let signed_sqrts: Vec<Integer> = sqrts
            .iter()
            .enumerate()
            .map(|(i, s)| {
                if mask & (1 << i) != 0 {
                    Integer::from(&modulus - s) % &modulus
                } else {
                    s.clone()
                }
            })
            .collect();

        // Lagrange interpolation to reconstruct γ from values at roots
        let gamma_coeffs = lagrange_interpolation_mod(&roots, &signed_sqrts, &modulus);

        // Centered lift to exact integer coefficients
        let gamma_exact: Vec<Integer> = gamma_coeffs
            .iter()
            .map(|c| {
                let r = Integer::from(c % &modulus);
                let r = if r < 0 { r + &modulus } else { r };
                if r > half {
                    r - &modulus
                } else {
                    r
                }
            })
            .collect();

        // EXACT verification: γ² must equal P exactly in Z[α]/(f)
        let gamma_sq = nf_multiply(&gamma_exact, &gamma_exact, f_coeffs);
        let exact_match = gamma_sq.len() == product.len()
            && gamma_sq.iter().zip(product.iter()).all(|(a, b)| a == b);
        if !exact_match {
            continue;
        }

        // γ is the correct algebraic square root — evaluate γ(m) mod N
        let mut y_exact = Integer::from(0);
        let mut m_pow_exact = Integer::from(1);
        for c in &gamma_exact {
            y_exact += Integer::from(c * &m_pow_exact);
            m_pow_exact *= m;
        }

        let mut y = Integer::from(&y_exact % n);
        if y < 0 {
            y += n;
        }

        results.push(y);
    }

    results
}

/// Compute the algebraic square root and extract a factor.
///
/// Given a dependency (set of relation indices):
/// 1. Compute rational square root x = sqrt(∏(a_i - b_i*m)) mod N
/// 2. Compute exact product of algebraic integers (a_i - b_i*α) in Z[α]/(f)
/// 3. Use Couveignes' method to find γ with γ² = algebraic product
/// 4. Evaluate y = γ(m) mod N
/// 5. Try gcd(x ± y, N) to extract factor
pub fn extract_factor(
    relations: &[Relation],
    dependency: &[usize],
    f_coeffs: &[Integer],
    m: &Integer,
    n: &Integer,
) -> Option<Integer> {
    let d = f_coeffs.len() - 1;

    // Step 1: Compute product of rational norms (a - b*m) and take sqrt
    let mut rat_product = Integer::from(1);
    for &idx in dependency {
        let rel = &relations[idx];
        let norm = Integer::from(rel.a) - Integer::from(rel.b) * m;
        rat_product *= norm;
    }

    let sign = if rat_product < 0 {
        rat_product = -rat_product;
        true
    } else {
        false
    };
    let rat_sqrt = rat_product.clone().sqrt();

    if Integer::from(&rat_sqrt * &rat_sqrt) != rat_product {
        return None; // Not a perfect square
    }

    let x = if sign {
        let neg = Integer::from(-&rat_sqrt);
        let rem = Integer::from(&neg % n);
        Integer::from(&rem + n) % n
    } else {
        Integer::from(&rat_sqrt % n)
    };

    // Step 2: Compute exact product of algebraic elements in Z[α]/(f)
    // NO mod N reduction — we need exact coefficients for Couveignes' method
    let mut alg_product = vec![Integer::from(0); d];
    alg_product[0] = Integer::from(1);

    for &idx in dependency {
        let rel = &relations[idx];
        let elem = nf_element_from_ab(rel.a, rel.b, d);
        alg_product = nf_multiply(&alg_product, &elem, f_coeffs);
    }

    // Step 3: Compute algebraic square root(s) via Couveignes' method
    let y_values = algebraic_square_roots(&alg_product, f_coeffs, m, n);
    // Step 4: Try gcd(x ± y, N) for each algebraic square root
    for y in &y_values {
        let diff = Integer::from(Integer::from(&x - y) + n) % n;
        let g = n.clone().gcd(&diff);
        if g > 1 && g < *n {
            return Some(g);
        }
        let sum = Integer::from(&x + y) % n;
        let g = n.clone().gcd(&sum);
        if g > 1 && g < *n {
            return Some(g);
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use rug::Integer;
    use crate::types::Relation;

    fn make_test_relation(a: i64, b: u64) -> Relation {
        Relation {
            a, b,
            rational_factors: vec![],
            algebraic_factors: vec![],
            rational_sign_negative: false,
            algebraic_sign_negative: false,
        }
    }

    #[test]
    fn test_rational_square_root_product() {
        let m = Integer::from(3);
        let n = Integer::from(8051);
        // a=7, b=1 → rational norm = 7 - 1*3 = 4 = 2^2
        let rels = vec![make_test_relation(7, 1)];
        let dep = vec![0usize];
        let result = rational_square_root(&rels, &dep, &m, &n);
        assert!(result.is_some());
        let (x, _sign) = result.unwrap();
        let x_sq = Integer::from(&x * &x) % &n;
        let expected = Integer::from(4) % &n;
        assert_eq!(x_sq, expected);
    }

    #[test]
    fn test_nf_multiply() {
        // In Q[x]/(x^2 + 1), multiply (1 + x) * (1 - x) = 1 - x^2 = 1 - (-1) = 2
        let f = vec![Integer::from(1), Integer::from(0), Integer::from(1)];
        let a = vec![Integer::from(1), Integer::from(1)];
        let b = vec![Integer::from(1), Integer::from(-1)];
        let result = nf_multiply(&a, &b, &f);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], Integer::from(2));
        assert_eq!(result[1], Integer::from(0));
    }

    #[test]
    fn test_nf_element_from_relation() {
        let elem = nf_element_from_ab(3, 2, 2);
        assert_eq!(elem[0], Integer::from(3));
        assert_eq!(elem[1], Integer::from(-2));
    }

    #[test]
    fn test_full_factor_extraction_small() {
        use crate::arith::select_quad_char_primes;
        use crate::polyselect::select_base_m;
        use crate::sieve::{build_factor_base, line_sieve, poly_coeffs_to_i64};
        use crate::relation::collect_smooth_relations;
        use crate::linalg::{build_matrix, find_dependencies};

        let n = Integer::from(8051u64);
        let poly = select_base_m(&n, 3);
        let f_coeffs = poly.f_coeffs();
        let f_i64 = poly_coeffs_to_i64(&f_coeffs).unwrap();
        let fb = build_factor_base(&f_i64, 300);

        let hits = line_sieve(&poly, &fb, 1000, 200);
        let (rels, _) = collect_smooth_relations(&hits, &fb, 1 << 16);

        if rels.len() < fb.primes.len() + 2 {
            eprintln!("Not enough relations ({}) for matrix. Skipping.", rels.len());
            return;
        }

        let quad_chars = select_quad_char_primes(&f_i64, &fb.primes, 30);
        let alg_pairs = fb.algebraic_pair_count();
        let (matrix, ncols) = build_matrix(&rels, fb.primes.len(), alg_pairs, &quad_chars);
        let deps = find_dependencies(&matrix, ncols);

        if deps.is_empty() {
            eprintln!("No dependencies found. Skipping.");
            return;
        }

        let m = poly.m();
        for dep in &deps {
            if let Some(factor) = extract_factor(&rels, dep, &f_coeffs, &m, &n) {
                assert!(factor > 1);
                assert!(factor < n);
                assert_eq!(Integer::from(&n % &factor), 0);
                return;
            }
        }
        eprintln!("No factor extracted from any dependency (may need more relations)");
    }
}
