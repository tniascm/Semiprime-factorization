use crate::sieve::SieveHit;
use crate::types::{FactorBase, Relation};

/// Trial divide n by the factor base primes. Returns ([(prime_index, exponent)], remainder).
pub fn trial_divide(mut n: u64, primes: &[u64]) -> (Vec<(u32, u8)>, u64) {
    let mut factors = Vec::new();
    for (i, &p) in primes.iter().enumerate() {
        if n == 1 {
            break;
        }
        let mut exp = 0u8;
        while n % p == 0 {
            n /= p;
            exp += 1;
        }
        if exp > 0 {
            factors.push((i as u32, exp));
        }
    }
    (factors, n)
}

/// Compute v_p(|val|) — the p-adic valuation of an integer.
fn p_adic_val(mut val: u64, p: u64) -> u8 {
    if val == 0 || p < 2 {
        return 0;
    }
    let mut e = 0u8;
    while val % p == 0 {
        val /= p;
        e += 1;
    }
    e
}

/// Collect smooth relations from sieve hits by trial division.
///
/// For each hit, trial-divide both rational and algebraic norms by factor base primes.
/// Algebraic factors are tracked per (prime, root) pair — one entry per degree-1 prime
/// ideal — to ensure the GF(2) matrix correctly constrains individual ideal exponents.
/// Relations where per-root exponents don't fully account for the norm (indicating
/// higher-degree or ramified ideals) are rejected to guarantee matrix correctness.
///
/// Returns (full_relations, partial_relation_count).
pub fn collect_smooth_relations(
    hits: &[SieveHit],
    fb: &FactorBase,
    large_prime_bound: u64,
) -> (Vec<Relation>, usize) {
    let mut relations = Vec::new();
    let mut partial_count = 0;

    for hit in hits {
        let rat_sign_neg = hit.rational_norm < 0;
        let rat_abs = hit.rational_norm.clone().abs().to_u64().unwrap_or(u64::MAX);
        if rat_abs == u64::MAX || rat_abs == 0 {
            continue;
        }
        let (rat_factors, rat_remainder) = trial_divide(rat_abs, &fb.primes);

        let alg_sign_neg = hit.algebraic_norm < 0;
        let alg_abs = hit.algebraic_norm.clone().abs().to_u64().unwrap_or(u64::MAX);
        if alg_abs == u64::MAX || alg_abs == 0 {
            continue;
        }
        let (alg_prime_factors, alg_remainder) = trial_divide(alg_abs, &fb.primes);

        let is_smooth = rat_remainder == 1 && alg_remainder == 1;
        if !is_smooth {
            if rat_remainder <= large_prime_bound && alg_remainder <= large_prime_bound {
                partial_count += 1;
            }
            continue;
        }

        // Decompose algebraic exponents per (prime, root) pair.
        // For each prime p dividing the norm with roots r_1,...,r_k of f mod p:
        //   v_{(p,r_i)}(a - b*alpha) = v_p(a - b*r_i)
        // Only accept relations where per-root exponents fully account for
        // the total exponent. This guarantees all algebraic factors come from
        // degree-1 ideals, which the GF(2) matrix tracks correctly.
        // Relations involving higher-degree or ramified ideals are rejected
        // to avoid ideal-tracking errors.
        let mut alg_pair_factors: Vec<(u32, u8)> = Vec::new();
        let mut valid = true;

        for &(prime_idx, total_exp) in &alg_prime_factors {
            let p = fb.primes[prime_idx as usize];
            let roots = &fb.algebraic_roots[prime_idx as usize];
            let pair_base = fb.pair_offset(prime_idx as usize);

            let mut root_exp_sum = 0u8;
            for (root_idx, &r) in roots.iter().enumerate() {
                // Compute v_p(|a - b*r|) using i128 to avoid overflow
                let val_i128 = hit.a as i128 - hit.b as i128 * r as i128;
                let val_abs = val_i128.unsigned_abs() as u64;
                let e = p_adic_val(val_abs, p);
                if e > 0 {
                    alg_pair_factors.push(((pair_base + root_idx) as u32, e));
                    root_exp_sum = root_exp_sum.saturating_add(e);
                }
            }

            // Reject if degree-1 ideals don't fully explain the exponent
            if root_exp_sum != total_exp {
                valid = false;
                break;
            }
        }

        if !valid {
            continue;
        }

        relations.push(Relation {
            a: hit.a,
            b: hit.b,
            rational_factors: rat_factors,
            algebraic_factors: alg_pair_factors,
            rational_sign_negative: rat_sign_neg,
            algebraic_sign_negative: alg_sign_neg,
        });
    }

    (relations, partial_count)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rug::Integer;

    #[test]
    fn test_trial_divide() {
        let primes = vec![2, 3, 5, 7, 11, 13];
        let n = 2u64 * 3 * 5 * 7;
        let (exponents, remainder) = trial_divide(n, &primes);
        assert_eq!(remainder, 1);
        assert_eq!(exponents, vec![(0, 1), (1, 1), (2, 1), (3, 1)]);
    }

    #[test]
    fn test_trial_divide_with_remainder() {
        let primes = vec![2, 3, 5];
        let n = 2u64 * 3 * 17;
        let (exponents, remainder) = trial_divide(n, &primes);
        assert_eq!(remainder, 17);
        assert_eq!(exponents, vec![(0, 1), (1, 1)]);
    }

    #[test]
    fn test_trial_divide_large_prime() {
        let primes = vec![2, 3, 5, 7];
        let n = 4u64 * 9 * 23;
        let (exponents, remainder) = trial_divide(n, &primes);
        assert_eq!(remainder, 23);
        assert_eq!(exponents, vec![(0, 2), (1, 2)]);
    }

    #[test]
    fn test_collect_smooth_relations() {
        use crate::polyselect::select_base_m;
        use crate::sieve::{build_factor_base, line_sieve, poly_coeffs_to_i64};

        let n = Integer::from(8051u64);
        let poly = select_base_m(&n, 3);
        let f_coeffs = poly.f_coeffs();
        let f_i64 = poly_coeffs_to_i64(&f_coeffs).unwrap();
        let fb = build_factor_base(&f_i64, 200);

        let hits = line_sieve(&poly, &fb, 500, 100);
        let (relations, _partial_count) = collect_smooth_relations(
            &hits, &fb, 1 << 16,
        );
        assert!(!relations.is_empty(), "Should find smooth relations for 8051");
    }
}
