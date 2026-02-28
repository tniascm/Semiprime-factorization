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

/// Collect smooth relations from sieve hits by trial division.
///
/// For each hit, trial-divide both rational and algebraic norms by factor base primes.
/// If the remainder is 1, it's a full relation.
/// If the remainder fits within the large prime bound, it's a partial (counted but not used in M1).
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
        let rat_abs = hit.rational_norm.to_u64().unwrap_or(u64::MAX);
        if rat_abs == u64::MAX || rat_abs == 0 {
            continue;
        }
        let (rat_factors, rat_remainder) = trial_divide(rat_abs, &fb.primes);

        let alg_abs = hit.algebraic_norm.to_u64().unwrap_or(u64::MAX);
        if alg_abs == u64::MAX || alg_abs == 0 {
            continue;
        }
        let (alg_factors, alg_remainder) = trial_divide(alg_abs, &fb.primes);

        if rat_remainder == 1 && alg_remainder == 1 {
            relations.push(Relation {
                a: hit.a,
                b: hit.b,
                rational_factors: rat_factors,
                algebraic_factors: alg_factors,
                rational_sign_negative: hit.rational_norm < 0,
                algebraic_sign_negative: hit.algebraic_norm < 0,
            });
        } else if rat_remainder <= large_prime_bound && alg_remainder <= large_prime_bound {
            partial_count += 1;
        }
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
