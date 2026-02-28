use crate::arith::{find_polynomial_roots_mod_p, sieve_primes};
use crate::types::{FactorBase, PolynomialPair};
use rug::Integer;

/// Evaluate polynomial with i64 coefficients at x mod p.
/// Coefficients are [c0, c1, ..., cd] (low-degree first).
fn eval_poly_mod_i64(coeffs: &[i64], x: u64, p: u64) -> u64 {
    crate::arith::eval_poly_mod(coeffs, x, p)
}

/// Build the factor base: primes up to bound with roots of f(x) mod p.
/// Only includes primes where f has at least one root mod p.
pub fn build_factor_base(f_coeffs_i64: &[i64], bound: u64) -> FactorBase {
    let all_primes = sieve_primes(bound);
    let mut primes = Vec::new();
    let mut algebraic_roots = Vec::new();
    let mut log_p_vals = Vec::new();

    for &p in &all_primes {
        let roots = find_polynomial_roots_mod_p(f_coeffs_i64, p);
        if !roots.is_empty() {
            log_p_vals.push(((p as f64).log2() * 20.0).round() as u8);
            primes.push(p);
            algebraic_roots.push(roots);
        }
    }

    FactorBase {
        primes,
        algebraic_roots,
        log_p: log_p_vals,
    }
}

/// Convert rug::Integer polynomial coefficients to i64 for sieve operations.
/// Returns None if any coefficient doesn't fit in i64.
pub fn poly_coeffs_to_i64(coeffs: &[Integer]) -> Option<Vec<i64>> {
    coeffs.iter().map(|c| c.to_i64()).collect()
}

/// A sieve hit: candidate (a, b) pair where norms may be smooth.
#[derive(Debug, Clone)]
pub struct SieveHit {
    pub a: i64,
    pub b: u64,
    pub rational_norm: Integer,
    pub algebraic_norm: Integer,
}

/// Line sieve: for each b in [1, max_b], sweep a over [-sieve_a, sieve_a].
///
/// Accumulates log(p) for factor base primes dividing the rational and algebraic norms.
/// Candidates where the accumulated log is close to log(norm) are potential smooth relations.
///
/// Returns candidate (a, b) pairs with their norms for trial division.
pub fn line_sieve(
    poly: &PolynomialPair,
    fb: &FactorBase,
    sieve_a: u64,
    max_b: u64,
) -> Vec<SieveHit> {
    let m = poly.m();
    let f_coeffs = poly.f_coeffs();
    let _f_i64 = match poly_coeffs_to_i64(&f_coeffs) {
        Some(v) => v,
        None => return vec![],
    };
    let sieve_a_i64 = sieve_a as i64;
    let sieve_len = (2 * sieve_a + 1) as usize;

    let mut hits = Vec::new();

    for b in 1..=max_b {
        let mut rat_sieve = vec![0u16; sieve_len];
        let mut alg_sieve = vec![0u16; sieve_len];

        for (i, &p) in fb.primes.iter().enumerate() {
            let log_val = fb.log_p[i] as u16;
            let p_i64 = p as i64;

            // Rational: a ≡ -b*m (mod p)
            let bm_mod_p = ((b as i128 * m.to_i128().unwrap_or(0)) % p as i128).unsigned_abs() as u64;
            let rat_start = (p - (bm_mod_p % p)) % p;
            let offset = ((rat_start as i64 + sieve_a_i64) % p_i64 + p_i64) % p_i64;
            let mut idx = offset as usize;
            while idx < sieve_len {
                rat_sieve[idx] += log_val;
                idx += p as usize;
            }

            // Algebraic side: for each root r of f mod p, a ≡ b*r (mod p)
            for &r in &fb.algebraic_roots[i] {
                let br_mod_p = ((b as u128 * r as u128) % p as u128) as u64;
                let alg_offset = ((br_mod_p as i64 + sieve_a_i64) % p_i64 + p_i64) % p_i64;
                let mut idx = alg_offset as usize;
                while idx < sieve_len {
                    alg_sieve[idx] += log_val;
                    idx += p as usize;
                }
            }
        }

        // Identify survivors
        for idx in 0..sieve_len {
            let a = idx as i64 - sieve_a_i64;
            if a == 0 {
                continue;
            }

            let rat_acc = rat_sieve[idx];
            let alg_acc = alg_sieve[idx];

            let rational_norm = poly.eval_g(a, b);
            let algebraic_norm = poly.eval_f_homogeneous(a, b);

            if rational_norm == 0 || algebraic_norm == 0 {
                continue;
            }

            let rat_bits = rational_norm.significant_bits() as u16;
            let alg_bits = algebraic_norm.significant_bits() as u16;

            // Threshold: accumulated log should be at least ~50% of norm bits
            let rat_threshold = (rat_bits as f64 * 20.0 * 0.5) as u16;
            let alg_threshold = (alg_bits as f64 * 20.0 * 0.5) as u16;

            if rat_acc >= rat_threshold && alg_acc >= alg_threshold {
                hits.push(SieveHit {
                    a,
                    b,
                    rational_norm: rational_norm.abs(),
                    algebraic_norm: algebraic_norm.abs(),
                });
            }
        }
    }

    hits
}

#[cfg(test)]
mod tests {
    use super::*;
    use rug::Integer;

    #[test]
    fn test_build_factor_base() {
        let f_coeffs = vec![1i64, 0, 1]; // x^2 + 1
        let fb = build_factor_base(&f_coeffs, 30);
        assert!(fb.primes.contains(&2));
        assert!(fb.primes.contains(&5));
        for (i, p) in fb.primes.iter().enumerate() {
            for &r in &fb.algebraic_roots[i] {
                let val = eval_poly_mod_i64(&[1, 0, 1], r, *p);
                assert_eq!(val, 0, "Root {} of f mod {} should give 0", r, p);
            }
        }
    }

    #[test]
    fn test_log_p_values() {
        let f_coeffs = vec![1i64, 0, 1];
        let fb = build_factor_base(&f_coeffs, 100);
        assert!(fb.log_p[0] > 0);
        let idx_97 = fb.primes.iter().position(|&p| p == 97);
        if let Some(idx) = idx_97 {
            assert!(fb.log_p[idx] > fb.log_p[0]);
        }
    }

    #[test]
    fn test_line_sieve_finds_smooth() {
        use crate::polyselect::select_base_m;
        let n = Integer::from(8051);
        let poly = select_base_m(&n, 3);
        let f_coeffs = poly.f_coeffs();
        let f_i64 = poly_coeffs_to_i64(&f_coeffs).unwrap();
        let fb = build_factor_base(&f_i64, 200);

        let hits = line_sieve(&poly, &fb, 500, 100);
        assert!(!hits.is_empty(), "Line sieve should find some candidate pairs");
    }
}
