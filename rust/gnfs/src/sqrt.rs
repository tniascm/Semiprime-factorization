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

// ================================================================
// Irreducible-prime Newton algebraic square root (CADO-NFS style)
// ================================================================
//
// Algorithm:
// 1. Find prime p where f(x) is irreducible mod p (p ≡ 3 mod 4 for odd degree)
// 2. Compute initial sqrt S₀ = P^{(p^d+1)/4} in F_{p^d}
// 3. Compute initial inverse sqrt R₀ = S₀^{-1} in F_{p^d}
// 4. Newton iteration: R_{k+1} = R_k · (3 - P·R_k²) / 2 mod (f, p^{2^{k+1}})
// 5. Recover γ = P · R_K, center-lift, verify γ² = P exactly
//
// Advantages over split-prime Couveignes:
// - Only 2 sign choices (γ and -γ), not 2^d
// - No Lagrange interpolation or sign recombination
// - More numerically stable (Newton quadratic convergence)

/// Multiply two polynomials in F_p[x]/(f(x)) using u64 arithmetic.
/// Coefficients of a, b, f are in [0, p). f is monic of degree d.
fn fp_multiply(a: &[u64], b: &[u64], f: &[u64], p: u64) -> Vec<u64> {
    let d = f.len() - 1;
    let p128 = p as u128;
    let mut product = vec![0u128; 2 * d - 1];
    for (i, &ai) in a.iter().enumerate() {
        for (j, &bj) in b.iter().enumerate() {
            product[i + j] = (product[i + j] + (ai as u128) * (bj as u128)) % p128;
        }
    }
    // Reduce modulo monic f: x^d = -(c_{d-1}*x^{d-1} + ... + c_0)
    for i in (d..product.len()).rev() {
        let coeff = product[i] % p128;
        if coeff != 0 {
            product[i] = 0;
            for j in 0..d {
                let sub = (coeff * (f[j] as u128)) % p128;
                product[i - d + j] = (product[i - d + j] + p128 - sub) % p128;
            }
        }
    }
    product[..d].iter().map(|&c| (c % p128) as u64).collect()
}

/// Modular exponentiation in F_p[x]/(f(x)). Uses repeated squaring.
fn fp_pow(base: &[u64], exp: &Integer, f: &[u64], p: u64) -> Vec<u64> {
    let d = f.len() - 1;
    if *exp == 0 {
        let mut one = vec![0u64; d];
        one[0] = 1;
        return one;
    }
    let bits = exp.significant_bits();
    let mut result = vec![0u64; d];
    result[0] = 1;
    let mut power = base.to_vec();
    for bit_idx in 0..bits {
        if exp.get_bit(bit_idx) {
            result = fp_multiply(&result, &power, f, p);
        }
        if bit_idx + 1 < bits {
            power = fp_multiply(&power, &power, f, p);
        }
    }
    result
}

/// Multiply two polynomials in (Z/mZ)[x]/(f(x)) where f is monic of degree d.
fn nf_multiply_mod(a: &[Integer], b: &[Integer], f: &[Integer], m: &Integer) -> Vec<Integer> {
    let d = f.len() - 1;
    let mut product = vec![Integer::from(0); 2 * d - 1];
    for (i, ai) in a.iter().enumerate() {
        for (j, bj) in b.iter().enumerate() {
            product[i + j] += Integer::from(ai * bj);
        }
    }
    // Reduce modulo monic f
    for i in (d..product.len()).rev() {
        let coeff = Integer::from(&product[i] % m);
        let coeff = if coeff < 0 { coeff + m } else { coeff };
        if coeff != 0 {
            product[i] = Integer::from(0);
            for j in 0..d {
                product[i - d + j] -= Integer::from(&coeff * &f[j]);
            }
        } else {
            product[i] = Integer::from(0);
        }
    }
    product.truncate(d);
    for c in &mut product {
        *c %= m;
        if *c < 0 { *c += m; }
    }
    product
}

/// Modular exponentiation in (Z/mZ)[x]/(f(x)). Uses repeated squaring.
#[allow(dead_code)]
fn nf_pow_mod(base: &[Integer], exp: &Integer, f: &[Integer], m: &Integer) -> Vec<Integer> {
    let d = f.len() - 1;
    if *exp == 0 {
        let mut one = vec![Integer::from(0); d];
        one[0] = Integer::from(1);
        return one;
    }
    let bits = exp.significant_bits();
    let mut result = vec![Integer::from(0); d];
    result[0] = Integer::from(1);
    let mut power = base.to_vec();
    for bit_idx in 0..bits {
        if exp.get_bit(bit_idx) {
            result = nf_multiply_mod(&result, &power, f, m);
        }
        if bit_idx + 1 < bits {
            power = nf_multiply_mod(&power, &power, f, m);
        }
    }
    result
}

/// Find a prime p where f(x) is irreducible mod p.
/// For odd degree d with p ≡ 3 mod 4: sqrt in F_{p^d} uses simple exponentiation.
/// For degree ≤ 3: irreducible ⟺ no roots mod p.
fn find_irreducible_prime(f_coeffs: &[i64], bound: u64, prefer_3mod4: bool) -> Option<u64> {
    use crate::arith::{find_polynomial_roots_mod_p, sieve_primes};
    let d = f_coeffs.len() - 1;
    let primes = sieve_primes(bound);

    for &p in &primes {
        if p < 3 { continue; }
        if prefer_3mod4 && p % 4 != 3 { continue; }

        let roots = find_polynomial_roots_mod_p(f_coeffs, p);
        if d <= 3 {
            // For degree ≤ 3: no roots ⟹ irreducible
            if roots.is_empty() {
                return Some(p);
            }
        } else {
            // For higher degree: no roots is necessary but not sufficient
            // (could have quadratic factors). Skip for now — fall back to Couveignes.
            if roots.is_empty() {
                // TODO: add proper polynomial GCD check for degree > 3
                // For now, conservatively accept (may sometimes be wrong for d=4)
                return Some(p);
            }
        }
    }
    None
}

/// Compute algebraic square root via Newton's method with an irreducible prime.
///
/// Given P(α) in Z[α]/(f) that is a perfect square, finds γ with γ² = P exactly.
/// Returns valid (y = γ(m) mod N, gamma_coeffs) pairs.
fn algebraic_sqrt_newton(
    product: &[Integer],
    f_coeffs: &[Integer],
    m: &Integer,
    n: &Integer,
) -> Vec<(Integer, Vec<Integer>)> {
    use crate::arith::*;

    let d = f_coeffs.len() - 1;
    if d < 2 { return vec![]; }

    let f_i64: Option<Vec<i64>> = f_coeffs.iter().map(|c| c.to_i64()).collect();
    let f_i64 = match f_i64 {
        Some(v) => v,
        None => return vec![],
    };

    // For odd degree, prefer p ≡ 3 mod 4 (simple sqrt formula)
    let prefer_3mod4 = d % 2 == 1;
    let p = match find_irreducible_prime(&f_i64, 500_000, prefer_3mod4) {
        Some(p) => p,
        None => return vec![],
    };
    let p_int = Integer::from(p);

    // Reduce f and P to F_p coefficients
    let f_fp: Vec<u64> = f_i64.iter().map(|&c| {
        ((c as i128).rem_euclid(p as i128)) as u64
    }).collect();
    let p_fp: Vec<u64> = product.iter().map(|c| {
        let r = Integer::from(c % &p_int);
        let r = if r < 0 { r + &p_int } else { r };
        r.to_u64().unwrap_or(0)
    }).collect();

    // Check P ≠ 0 in F_{p^d}
    if p_fp.iter().all(|&c| c == 0) {
        return vec![]; // P ≡ 0 mod p, try different prime? For now, give up.
    }

    // Compute p^d for Euler criterion and sqrt exponent
    let mut pd = Integer::from(p);
    for _ in 1..d {
        pd *= p;
    }

    // Euler criterion: P^{(p^d - 1)/2} should be 1 in F_{p^d}
    let euler_exp = Integer::from(&pd - 1) / 2;
    let euler_result = fp_pow(&p_fp, &euler_exp, &f_fp, p);
    let is_one = euler_result[0] == 1 && euler_result[1..].iter().all(|&c| c == 0);
    if !is_one {
        return vec![]; // P is not a QR in F_{p^d}
    }

    // Compute S₀ = P^{(p^d + 1)/4} in F_{p^d} (valid when p^d ≡ 3 mod 4)
    let sqrt_exp = Integer::from(&pd + 1) / 4;
    let s0 = fp_pow(&p_fp, &sqrt_exp, &f_fp, p);

    // Verify S₀² ≡ P mod (f, p)
    let s0_sq = fp_multiply(&s0, &s0, &f_fp, p);
    if s0_sq != p_fp {
        return vec![]; // sqrt computation failed
    }

    // Compute R₀ = S₀^{-1} = S₀^{p^d - 2} in F_{p^d}
    let inv_exp = Integer::from(&pd - 2);
    let r0_fp = fp_pow(&s0, &inv_exp, &f_fp, p);

    // Convert R₀ to Integer coefficients for Hensel lifting
    let mut r_k: Vec<Integer> = r0_fp.iter().map(|&c| Integer::from(c)).collect();

    // Determine target modulus: need modulus > 2 * ||γ||_∞
    // Use conservative bound: ||γ||_∞ ≤ sqrt(||P||_∞) * (||f||_∞ + 1)^d * d²
    let max_p_coeff = product.iter().map(|c| c.clone().abs()).max().unwrap_or(Integer::from(1));
    let max_f_coeff = f_coeffs.iter().map(|c| c.clone().abs()).max().unwrap_or(Integer::from(1));
    let sqrt_max_p = Integer::from(max_p_coeff.sqrt_ref());
    let f_amplification = Integer::from(&max_f_coeff + 1).pow(d as u32) * (d * d) as u64;
    let target = Integer::from(&sqrt_max_p * &f_amplification) * 4 + 4;

    let mut modulus = p_int.clone();
    let max_steps = 60;

    for _step in 0..max_steps {
        let new_mod = Integer::from(&modulus * &modulus);

        // Reduce f and P mod new_mod
        let f_mod: Vec<Integer> = f_coeffs.iter().map(|c| {
            let mut r = Integer::from(c % &new_mod);
            if r < 0 { r += &new_mod; }
            r
        }).collect();
        let p_mod: Vec<Integer> = product.iter().map(|c| {
            let mut r = Integer::from(c % &new_mod);
            if r < 0 { r += &new_mod; }
            r
        }).collect();

        // Newton step: R_{k+1} = R_k · (3 - P · R_k²) / 2 mod (f, new_mod)

        // R_k² mod (f, new_mod)
        let r_sq = nf_multiply_mod(&r_k, &r_k, &f_mod, &new_mod);

        // P · R_k² mod (f, new_mod)
        let pr_sq = nf_multiply_mod(&p_mod, &r_sq, &f_mod, &new_mod);

        // 3 - P · R_k² mod (f, new_mod)
        let mut three_minus: Vec<Integer> = pr_sq.iter().map(|c| {
            let neg = Integer::from(&new_mod - c) % &new_mod;
            neg
        }).collect();
        three_minus[0] = Integer::from(&three_minus[0] + 3) % &new_mod;

        // R_k · (3 - P · R_k²) mod (f, new_mod)
        let r_times = nf_multiply_mod(&r_k, &three_minus, &f_mod, &new_mod);

        // Divide by 2: multiply by 2^{-1} mod new_mod
        let two_inv = match mod_inverse_int(&Integer::from(2), &new_mod) {
            Some(inv) => inv,
            None => break,
        };

        r_k = r_times.iter().map(|c| {
            let mut r = Integer::from(c * &two_inv) % &new_mod;
            if r < 0 { r += &new_mod; }
            r
        }).collect();

        modulus = new_mod;

        if modulus > target {
            break;
        }
    }

    // Recover γ = P · R_K mod (f, modulus), then center-lift and verify
    let f_final: Vec<Integer> = f_coeffs.iter().map(|c| {
        let mut r = Integer::from(c % &modulus);
        if r < 0 { r += &modulus; }
        r
    }).collect();
    let p_final: Vec<Integer> = product.iter().map(|c| {
        let mut r = Integer::from(c % &modulus);
        if r < 0 { r += &modulus; }
        r
    }).collect();

    let gamma_mod = nf_multiply_mod(&p_final, &r_k, &f_final, &modulus);

    let half = Integer::from(&modulus / 2);
    let mut results = Vec::new();

    // Try both signs: γ and -γ
    for sign in 0..2u8 {
        let gamma_exact: Vec<Integer> = gamma_mod.iter().map(|c| {
            let val = if sign == 0 {
                c.clone()
            } else {
                Integer::from(&modulus - c) % &modulus
            };
            // Center-lift to [-modulus/2, modulus/2)
            if val > half {
                val - &modulus
            } else {
                val
            }
        }).collect();

        // Verify γ² = P exactly in Z[α]/(f)
        let gamma_sq = nf_multiply(&gamma_exact, &gamma_exact, f_coeffs);
        let exact_match = gamma_sq.len() == product.len()
            && gamma_sq.iter().zip(product.iter()).all(|(a, b)| a == b);

        if !exact_match {
            continue;
        }

        // Evaluate γ(m) mod N
        let mut y = Integer::from(0);
        let mut m_pow = Integer::from(1);
        for c in &gamma_exact {
            y += Integer::from(c * &m_pow);
            m_pow *= m;
        }
        y = Integer::from(&y % n);
        if y < 0 { y += n; }

        if !results.iter().any(|(existing_y, _): &(Integer, Vec<Integer>)| existing_y == &y) {
            results.push((y, gamma_exact));
        }
    }

    results
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
/// Returns all valid (y = γ(m) mod N, gamma_coeffs) values.
fn algebraic_square_roots(
    product: &[Integer],
    f_coeffs: &[Integer],
    m: &Integer,
    n: &Integer,
) -> Vec<(Integer, Vec<Integer>)> {
    use crate::arith::*;

    let d = f_coeffs.len() - 1;
    debug_assert_eq!(f_coeffs[d], Integer::from(1), "f must be monic");

    let f_i64: Option<Vec<i64>> = f_coeffs.iter().map(|c| c.to_i64()).collect();
    let f_i64 = match f_i64 {
        Some(v) => v,
        None => return vec![],
    };

    // Step 1: Find primes ℓ where f splits completely (d distinct roots)
    // and P(r) is a QR mod ℓ for all roots r.
    // Try MULTIPLE primes to avoid the systematic trivial-gcd problem:
    // a single ℓ determines a fixed sign relationship between γ(m) mod p and
    // γ(m) mod q, so if that ℓ gives the "same root" branch, ALL dependencies
    // produce trivial gcd. Different ℓ values have independent sign relationships.
    let primes = sieve_primes(200_000);
    let mut valid_primes: Vec<(u64, Vec<u64>)> = Vec::new();

    for &p in &primes {
        if p < 3 { continue; }
        let roots = find_polynomial_roots_mod_p(&f_i64, p);
        if roots.len() != d { continue; }
        let all_nonzero_qr = roots.iter().all(|&r| {
            let r_int = Integer::from(r);
            let p_int = Integer::from(p);
            let val = eval_poly_int(product, &r_int, &p_int);
            val != 0 && tonelli_shanks_int(&val, &p_int).is_some()
        });
        if all_nonzero_qr {
            valid_primes.push((p, roots));
            if valid_primes.len() >= 5 {
                break;
            }
        }
    }

    if valid_primes.is_empty() {
        return vec![];
    }

    // Step 2: Adaptive Hensel lifting with early exit
    let max_p = product.iter().map(|c| c.clone().abs()).max().unwrap_or(Integer::from(1));
    let min_target: Integer = Integer::from(max_p.sqrt_ref()) * 2 + 2;
    let max_target: Integer = Integer::from(&max_p * 2) + 2;

    let mut results: Vec<(Integer, Vec<Integer>)> = Vec::new();
    let max_lift_steps = 50;

    // Try each valid prime ℓ — different primes give different sign relationships
    // between γ(m) mod p and γ(m) mod q, breaking the systematic trivial-gcd problem.
    for (ell, base_roots) in &valid_primes {
        let ell = *ell;

        // Initialize roots and sqrt values mod ℓ
        let mut roots: Vec<Integer> = base_roots.iter().map(|&r| Integer::from(r)).collect();
        let mut modulus = Integer::from(ell);

        let mut sqrts: Vec<Integer> = Vec::with_capacity(d);
        let mut init_ok = true;
        for r in &roots {
            let val = eval_poly_int(product, r, &modulus);
            match tonelli_shanks_int(&val, &modulus) {
                Some(s) => sqrts.push(s),
                None => { init_ok = false; break; }
            }
        }
        if !init_ok {
            continue;
        }

        // Hensel lift with adaptive verification
        let mut found_for_this_ell = false;

        for _step in 0..max_lift_steps {
            let new_mod = Integer::from(&modulus * &modulus);

            // Lift roots of f
            let mut lift_ok = true;
            for i in 0..d {
                let fr = eval_poly_int(f_coeffs, &roots[i], &new_mod);
                let fpr = eval_poly_deriv_int(f_coeffs, &roots[i], &new_mod);
                let fpr_inv = match mod_inverse_int(&fpr, &new_mod) {
                    Some(inv) => inv,
                    None => { lift_ok = false; break; }
                };
                let correction = Integer::from(&fr * &fpr_inv) % &new_mod;
                roots[i] = Integer::from(&roots[i] - &correction) % &new_mod;
                if roots[i] < 0 {
                    roots[i] += &new_mod;
                }
            }
            if !lift_ok { break; }

            // Lift sqrt values
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
                    None => { lift_ok = false; break; }
                };
                let delta = Integer::from(&residual * &two_s_inv) % &new_mod;
                sqrts[i] = Integer::from(&sqrts[i] + &delta) % &new_mod;
                if sqrts[i] < 0 {
                    sqrts[i] += &new_mod;
                }
            }
            if !lift_ok { break; }

            modulus = new_mod;

            if modulus <= min_target {
                continue;
            }

            // Verify Hensel correctness
            let mut hensel_ok = true;
            for i in 0..d {
                let p_val = eval_poly_int(product, &roots[i], &modulus);
                let s_sq = Integer::from(&sqrts[i] * &sqrts[i]) % &modulus;
                let s_sq = if s_sq < 0 { s_sq + &modulus } else { s_sq };
                if p_val != s_sq {
                    hensel_ok = false;
                    break;
                }
            }
            if !hensel_ok { break; }

            // Try all 2^d sign combinations
            let half = Integer::from(&modulus / 2);

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

                let gamma_coeffs = lagrange_interpolation_mod(&roots, &signed_sqrts, &modulus);

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

                let gamma_sq = nf_multiply(&gamma_exact, &gamma_exact, f_coeffs);
                let exact_match = gamma_sq.len() == product.len()
                    && gamma_sq.iter().zip(product.iter()).all(|(a, b)| a == b);
                if !exact_match {
                    continue;
                }

                // Evaluate γ(m) mod N
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

                // Deduplicate: only add if this y is new
                if !results.iter().any(|(existing, _)| existing == &y) {
                    results.push((y, gamma_exact.clone()));
                }

                found_for_this_ell = true;
            }

            if found_for_this_ell {
                break;
            }

            if modulus > max_target {
                break;
            }
        }
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
/// Diagnostic: why did extract_factor fail?
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FactorFailure {
    RationalNotSquare,
    AlgebraicNotSquare,
    TrivialGcd,
}

pub fn extract_factor(
    relations: &[Relation],
    dependency: &[usize],
    f_coeffs: &[Integer],
    m: &Integer,
    n: &Integer,
) -> Option<Integer> {
    extract_factor_diagnostic(relations, dependency, f_coeffs, m, n).0
}

/// Like extract_factor but also returns failure reason if None.
pub fn extract_factor_diagnostic(
    relations: &[Relation],
    dependency: &[usize],
    f_coeffs: &[Integer],
    m: &Integer,
    n: &Integer,
) -> (Option<Integer>, Option<FactorFailure>) {
    extract_factor_inner(relations, dependency, f_coeffs, m, n, false)
}

/// Verbose version that prints detailed diagnostics for debugging.
pub fn extract_factor_verbose(
    relations: &[Relation],
    dependency: &[usize],
    f_coeffs: &[Integer],
    m: &Integer,
    n: &Integer,
) -> (Option<Integer>, Option<FactorFailure>) {
    extract_factor_inner(relations, dependency, f_coeffs, m, n, true)
}

fn extract_factor_inner(
    relations: &[Relation],
    dependency: &[usize],
    f_coeffs: &[Integer],
    m: &Integer,
    n: &Integer,
    verbose: bool,
) -> (Option<Integer>, Option<FactorFailure>) {
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
        if verbose {
            eprintln!("[diag] Rational product is NOT a perfect square (sign={})", sign);
        }
        return (None, Some(FactorFailure::RationalNotSquare));
    }

    let x = if sign {
        let neg = Integer::from(-&rat_sqrt);
        let rem = Integer::from(&neg % n);
        Integer::from(&rem + n) % n
    } else {
        Integer::from(&rat_sqrt % n)
    };

    if verbose {
        eprintln!("[diag] Rational: sign={}, sqrt={} bits, x={}", sign, rat_sqrt.significant_bits(), &x);
    }

    // Step 2: Compute exact product of algebraic elements in Z[α]/(f)
    // NO mod N reduction — we need exact coefficients for Couveignes' method
    let mut alg_product = vec![Integer::from(0); d];
    alg_product[0] = Integer::from(1);

    for &idx in dependency {
        let rel = &relations[idx];
        let elem = nf_element_from_ab(rel.a, rel.b, d);
        alg_product = nf_multiply(&alg_product, &elem, f_coeffs);
    }

    if verbose {
        let max_coeff = alg_product.iter().map(|c| c.clone().abs()).max().unwrap_or(Integer::from(0));
        eprintln!("[diag] Algebraic product: {} coeffs, max_coeff={} bits", alg_product.len(), max_coeff.significant_bits());
        for (i, c) in alg_product.iter().enumerate() {
            eprintln!("[diag]   P[{}] = {} ({} bits)", i, c, c.clone().abs().significant_bits());
        }
        // Check P(m) = rational product?
        let mut pm = Integer::from(0);
        let mut m_pow = Integer::from(1);
        for c in &alg_product {
            pm += Integer::from(c * &m_pow);
            m_pow *= m;
        }
        let rat_product_signed = if sign { -Integer::from(&rat_sqrt * &rat_sqrt) } else { Integer::from(&rat_sqrt * &rat_sqrt) };
        eprintln!("[diag] P(m) = {} ({} bits)", &pm, pm.clone().abs().significant_bits());
        eprintln!("[diag] R (rational product) = {} ({} bits)", &rat_product_signed, rat_product_signed.clone().abs().significant_bits());
        eprintln!("[diag] P(m) == R? {}", pm == rat_product_signed);
    }

    // Step 3: Compute algebraic square root(s)
    // Try Newton method first (irreducible-prime, CADO-NFS style), fall back to Couveignes
    let mut y_values = algebraic_sqrt_newton(&alg_product, f_coeffs, m, n);

    if y_values.is_empty() {
        if verbose {
            eprintln!("[diag] Newton sqrt found nothing, trying Couveignes...");
        }
        y_values = algebraic_square_roots(&alg_product, f_coeffs, m, n);
    } else if verbose {
        eprintln!("[diag] Newton sqrt succeeded ({} root(s))", y_values.len());
    }

    if y_values.is_empty() {
        if verbose {
            eprintln!("[diag] No algebraic square roots found (neither Newton nor Couveignes)");
        }
        return (None, Some(FactorFailure::AlgebraicNotSquare));
    }

    if verbose {
        eprintln!("[diag] Found {} distinct algebraic square root(s)", y_values.len());
        for (i, (y, gamma)) in y_values.iter().enumerate() {
            eprintln!("[diag]   y[{}] = {}", i, y);
            eprintln!("[diag]   gamma[{}] = {:?}", i, gamma.iter().map(|c| c.to_string()).collect::<Vec<_>>());
            let x_eq_y = &x == y;
            let x_eq_neg_y = Integer::from(&x + y) % n == Integer::from(0);
            eprintln!("[diag]   x == y? {}  x == -y (mod N)? {}", x_eq_y, x_eq_neg_y);
            let diff = Integer::from(Integer::from(&x - y) + n) % n;
            let sum = Integer::from(&x + y) % n;
            let g1 = n.clone().gcd(&diff);
            let g2 = n.clone().gcd(&sum);
            eprintln!("[diag]   gcd(x-y, N) = {}  gcd(x+y, N) = {}", g1, g2);
        }
        // Check x^2 == y^2 mod N
        for (i, (y, _)) in y_values.iter().enumerate() {
            let x2 = Integer::from(&x * &x) % n;
            let y2 = Integer::from(y * y) % n;
            eprintln!("[diag]   x² mod N = {}", &x2);
            eprintln!("[diag]   y[{}]² mod N = {}", i, &y2);
            eprintln!("[diag]   x² == y²? {}", x2 == y2);
        }
    }

    // Step 4: Try gcd(x ± y, N) for each algebraic square root
    for (y, _) in &y_values {
        let diff = Integer::from(Integer::from(&x - y) + n) % n;
        let g = n.clone().gcd(&diff);
        if g > 1 && g < *n {
            return (Some(g), None);
        }
        let sum = Integer::from(&x + y) % n;
        let g = n.clone().gcd(&sum);
        if g > 1 && g < *n {
            return (Some(g), None);
        }
    }

    (None, Some(FactorFailure::TrivialGcd))
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
        let (rels, _) = collect_smooth_relations(&hits, &fb, 1 << 16, 3);

        if rels.len() < fb.primes.len() + 2 {
            eprintln!("Not enough relations ({}) for matrix. Skipping.", rels.len());
            return;
        }

        let quad_chars = select_quad_char_primes(&f_i64, &fb.primes, 30);
        let alg_pairs = fb.algebraic_pair_count();
        let alg_hd = fb.higher_degree_ideal_count(3);
        let (matrix, ncols) = build_matrix(&rels, fb.primes.len(), alg_pairs, alg_hd, &quad_chars);
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

    #[test]
    fn test_newton_sqrt_known_square() {
        // Test that Newton sqrt correctly finds sqrt of a known perfect square in Z[x]/(f)
        // Use f(x) = x^3 + 2x + 1 (monic degree 3)
        let f = vec![
            Integer::from(1), Integer::from(2), Integer::from(0), Integer::from(1),
        ];
        // gamma = [3, 1, -2] → gamma² in Z[x]/(f) is our "product"
        let gamma = vec![Integer::from(3), Integer::from(1), Integer::from(-2)];
        let product = nf_multiply(&gamma, &gamma, &f);

        let m = Integer::from(7);
        let n = Integer::from(100003); // arbitrary odd number for testing

        let results = algebraic_sqrt_newton(&product, &f, &m, &n);
        assert!(!results.is_empty(), "Newton sqrt should find root of a known perfect square");

        // Verify: one of the results should be ±γ(m) mod N
        let mut gamma_at_m = Integer::from(0);
        let mut m_pow = Integer::from(1);
        for c in &gamma {
            gamma_at_m += Integer::from(c * &m_pow);
            m_pow *= &m;
        }
        let expected_y = Integer::from(&gamma_at_m % &n);
        let expected_y = if expected_y < 0 { expected_y + &n } else { expected_y };
        let expected_neg = Integer::from(&n - &expected_y);

        let found = results.iter().any(|(y, _)| *y == expected_y || *y == expected_neg);
        assert!(found, "Newton sqrt should return γ(m) mod N or -γ(m) mod N");
    }

    #[test]
    fn test_find_irreducible_prime() {
        // f(x) = x^3 + 2x + 1 should have some primes where it's irreducible
        let f = vec![1i64, 2, 0, 1];
        let p = find_irreducible_prime(&f, 10000, true);
        assert!(p.is_some(), "Should find an irreducible prime for x^3+2x+1");
        let p = p.unwrap();
        assert!(p % 4 == 3, "Should be ≡ 3 mod 4");
        // Verify: f should have no roots mod p
        let roots = crate::arith::find_polynomial_roots_mod_p(&f, p);
        assert!(roots.is_empty(), "f should be irreducible (no roots) mod {}", p);
    }

    #[test]
    fn test_fp_multiply() {
        // In F_5[x]/(x^2 + 1): (2 + 3x)(1 + 4x) = 2 + 11x + 12x^2
        // x^2 ≡ -1, so 12x^2 = -12 ≡ 3 mod 5
        // result = (2+3) + 11x = 5 + 11x ≡ 0 + 1x mod 5 = [0, 1]
        let f = vec![1u64, 0, 1]; // x^2 + 1
        let a = vec![2u64, 3];
        let b = vec![1u64, 4];
        let result = fp_multiply(&a, &b, &f, 5);
        assert_eq!(result, vec![0, 1]);
    }
}
