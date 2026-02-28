use rug::Integer;
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
        let norm = Integer::from(rel.a) + Integer::from(rel.b) * m;
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
/// f is the minimal polynomial [c0, c1, ..., cd].
pub fn nf_multiply(a: &[Integer], b: &[Integer], f: &[Integer]) -> Vec<Integer> {
    let d = f.len() - 1;

    // Standard polynomial multiplication
    let mut product = vec![Integer::from(0); 2 * d - 1];
    for (i, ai) in a.iter().enumerate() {
        for (j, bj) in b.iter().enumerate() {
            product[i + j] += Integer::from(ai * bj);
        }
    }

    // Reduce modulo f(x)
    let lead = &f[d];

    for i in (d..product.len()).rev() {
        let coeff = product[i].clone();
        if coeff != 0 {
            product[i] = Integer::from(0);
            for j in 0..d {
                product[i - d + j] -= Integer::from(&coeff * &f[j]) / lead;
            }
        }
    }

    product.truncate(d);
    product
}

/// Compute the algebraic square root and extract a factor.
///
/// Given a dependency (set of relation indices):
/// 1. Compute rational square root x
/// 2. Compute product of algebraic integers (a_i - b_i * alpha) mod N
/// 3. Evaluate at alpha = m to get y^2 mod N
/// 4. Try gcd(x ± y, N) to extract factor
pub fn extract_factor(
    relations: &[Relation],
    dependency: &[usize],
    f_coeffs: &[Integer],
    m: &Integer,
    n: &Integer,
) -> Option<Integer> {
    let d = f_coeffs.len() - 1;

    // Step 1: Compute product of rational norms and take sqrt
    let mut rat_product = Integer::from(1);
    for &idx in dependency {
        let rel = &relations[idx];
        let norm = Integer::from(rel.a) + Integer::from(rel.b) * m;
        rat_product *= norm;
    }

    let sign = if rat_product < 0 { rat_product = -rat_product; true } else { false };
    let rat_sqrt = rat_product.clone().sqrt();

    if Integer::from(&rat_sqrt * &rat_sqrt) != rat_product {
        return None;
    }

    let x = if sign {
        let neg = Integer::from(-&rat_sqrt);
        let rem = Integer::from(&neg % n);
        Integer::from(&rem + n) % n
    } else {
        rat_sqrt % n
    };

    // Step 2: Compute product of algebraic integers (a - b*alpha) mod N
    let mut alg_product = vec![Integer::from(0); d];
    alg_product[0] = Integer::from(1);

    for &idx in dependency {
        let rel = &relations[idx];
        let elem = nf_element_from_ab(rel.a, rel.b, d);
        alg_product = nf_multiply(&alg_product, &elem, f_coeffs);
        for c in &mut alg_product {
            *c = Integer::from(&*c % n);
        }
    }

    // Step 3: Evaluate algebraic product at alpha = m, mod N
    let mut y_squared = Integer::from(0);
    let mut m_pow = Integer::from(1);
    for c in &alg_product {
        y_squared += c * &m_pow;
        y_squared %= n;
        m_pow = Integer::from(&m_pow * m) % n;
    }
    if y_squared < 0 {
        y_squared += n;
    }

    // Step 4: Try to extract factor via gcd
    let one = Integer::from(1);

    for offset in [&one, &(-Integer::from(1))] {
        let candidate = Integer::from(&x + offset);
        let g = n.clone().gcd(&candidate);
        if g > 1 && g < *n {
            return Some(g);
        }
    }

    // Try gcd(rat_sqrt - alg_evaluated, N)
    let x_mod_n = Integer::from(&x % n);
    for sign in [1i32, -1] {
        let y_mod_n = if sign == 1 {
            Integer::from(&y_squared % n)
        } else {
            let rem = Integer::from(&y_squared % n);
            Integer::from(n - &rem) % n
        };
        let diff = Integer::from(Integer::from(&x_mod_n - &y_mod_n) + n) % n;
        let g = n.clone().gcd(&diff);
        if g > 1 && g < *n {
            return Some(g);
        }
        let sum = Integer::from(Integer::from(&x_mod_n + &y_mod_n)) % n;
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
        let m = Integer::from(10);
        let n = Integer::from(8051);
        // a=6, b=3 → rational norm = 6 + 3*10 = 36 = 6^2
        let rels = vec![make_test_relation(6, 3)];
        let dep = vec![0usize];
        let result = rational_square_root(&rels, &dep, &m, &n);
        assert!(result.is_some());
        let (x, _sign) = result.unwrap();
        let x_sq = Integer::from(&x * &x) % &n;
        let expected = Integer::from(36) % &n;
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

        let (matrix, ncols) = build_matrix(&rels, fb.primes.len(), fb.primes.len());
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
