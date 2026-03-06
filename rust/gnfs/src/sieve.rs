use crate::arith::{find_polynomial_roots_mod_p, mod_inverse_u64, sieve_primes};
use crate::lattice::reduce_qlattice;
use crate::types::{FactorBase, PolynomialPair};
use rayon::prelude::*;
use rug::Integer;

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
    /// Special-q prime and root (q, r) if from lattice sieve. None for line sieve.
    pub special_q: Option<(u64, u64)>,
}

/// Line sieve: for each b in [1, max_b], sweep a over [-sieve_a, sieve_a].
///
/// Accumulates log(p) for factor base primes dividing the rational and algebraic norms.
/// Candidates where the accumulated log is close to log(norm) are potential smooth relations.
/// Uses rayon to parallelize across b values for multi-core speedup.
///
/// Returns candidate (a, b) pairs with their norms for trial division.
pub fn line_sieve(
    poly: &PolynomialPair,
    fb: &FactorBase,
    sieve_a: u64,
    max_b: u64,
) -> Vec<SieveHit> {
    let f_coeffs = poly.f_coeffs();
    if poly_coeffs_to_i64(&f_coeffs).is_none() {
        return vec![];
    }

    let m_i128 = poly.m().to_i128().unwrap_or(0);
    let sieve_a_i64 = sieve_a as i64;
    let sieve_len = (2 * sieve_a + 1) as usize;

    // Precompute f64 coefficients for fast approximate norm evaluation
    let f_f64: Vec<f64> = f_coeffs.iter().map(|c| c.to_f64()).collect();
    let degree = f_coeffs.len() - 1;

    // Parallel sieve: each b value is independent
    let hits: Vec<Vec<SieveHit>> = (1..=max_b)
        .into_par_iter()
        .map(|b| {
            let mut rat_sieve = vec![0u16; sieve_len];
            let mut alg_sieve = vec![0u16; sieve_len];

            for (i, &p) in fb.primes.iter().enumerate() {
                let log_val = fb.log_p[i] as u16;
                let p_i64 = p as i64;

                // Rational: a ≡ b*m (mod p)
                let bm_mod_p = ((b as i128 * m_i128) % p as i128).unsigned_abs() as u64;
                let rat_start = bm_mod_p % p;
                let offset = ((rat_start as i64 + sieve_a_i64) % p_i64 + p_i64) % p_i64;
                let mut idx = offset as usize;
                while idx < sieve_len {
                    rat_sieve[idx] = rat_sieve[idx].saturating_add(log_val);
                    idx += p as usize;
                }

                // Algebraic: for each root r, a ≡ b*r (mod p)
                for &r in &fb.algebraic_roots[i] {
                    let br_mod_p = ((b as u128 * r as u128) % p as u128) as u64;
                    let alg_offset = ((br_mod_p as i64 + sieve_a_i64) % p_i64 + p_i64) % p_i64;
                    let mut idx = alg_offset as usize;
                    while idx < sieve_len {
                        alg_sieve[idx] = alg_sieve[idx].saturating_add(log_val);
                        idx += p as usize;
                    }
                }
            }

            // Fast approximate threshold using f64 norms.
            // Only compute expensive rug::Integer norms for survivors.
            let b_f64 = b as f64;
            let m_f64 = m_i128 as f64;
            let mut local_hits = Vec::new();

            for idx in 0..sieve_len {
                let a = idx as i64 - sieve_a_i64;
                if a == 0 {
                    continue;
                }
                let rat_acc = rat_sieve[idx];
                let alg_acc = alg_sieve[idx];

                // Quick reject: minimum sieve score threshold
                if rat_acc < 20 || alg_acc < 20 {
                    continue;
                }

                // f64 approximate norms for threshold check
                let a_f64 = a as f64;
                let rat_norm_f64 = (a_f64 - b_f64 * m_f64).abs();
                let rat_bits_approx = rat_norm_f64.log2();
                let rat_threshold = ((rat_bits_approx * 20.0 * 0.5) as u16).saturating_sub(20);
                if rat_acc < rat_threshold {
                    continue;
                }

                // Algebraic norm via Horner in f64
                let x = a_f64 / b_f64;
                let mut alg_val = f_f64[degree];
                for k in (0..degree).rev() {
                    alg_val = alg_val * x + f_f64[k];
                }
                let alg_norm_f64 = alg_val.abs() * b_f64.powi(degree as i32);
                let alg_bits_approx = alg_norm_f64.log2();
                let alg_threshold = ((alg_bits_approx * 20.0 * 0.5) as u16).saturating_sub(20);
                if alg_acc < alg_threshold {
                    continue;
                }

                // Survivor: compute exact norms
                let rational_norm = poly.eval_g(a, b);
                let algebraic_norm = poly.eval_f_homogeneous(a, b);
                if rational_norm == 0 || algebraic_norm == 0 {
                    continue;
                }

                local_hits.push(SieveHit {
                    a,
                    b,
                    rational_norm,
                    algebraic_norm,
                    special_q: None,
                });
            }

            local_hits
        })
        .collect();

    hits.into_iter().flatten().collect()
}

/// Special-q lattice sieve: for each special-q prime q above the factor base bound,
/// compute a reduced 2D lattice and sieve in (i,j) coordinates.
///
/// Every lattice point automatically satisfies q | algebraic_norm(a,b), so the
/// effective algebraic norm for smoothness is F(a,b)/q — much smaller than the
/// line sieve's norms.
///
/// Parallelized across special-q values using rayon.
pub fn lattice_sieve(
    poly: &PolynomialPair,
    fb: &FactorBase,
    log_i: u32,
    qmin: u64,
    qmax: u64,
) -> Vec<SieveHit> {
    let f_coeffs = poly.f_coeffs();
    let f_i64 = match poly_coeffs_to_i64(&f_coeffs) {
        Some(v) => v,
        None => return vec![],
    };
    let m = poly.m();
    let degree = f_coeffs.len() - 1;
    let half_width = 1i64 << log_i;
    let sieve_width = (2 * half_width) as usize;
    let j_max = half_width as u64; // J = I

    // f64 coefficients for approximate norm evaluation (Horner's method)
    let f_f64: Vec<f64> = f_coeffs.iter().map(|c| c.to_f64()).collect();
    let m_f64 = m.to_f64();

    // Precompute m mod p for each factor base prime
    let m_mod_p: Vec<u64> = fb
        .primes
        .iter()
        .map(|&p| {
            let r = Integer::from(&m % Integer::from(p));
            let r = if r < 0 { r + Integer::from(p) } else { r };
            r.to_u64().unwrap_or(0)
        })
        .collect();

    // Enumerate special-q: primes in [qmin, qmax] with at least one root of f mod q
    let all_q_primes = sieve_primes(qmax);
    let special_qs: Vec<(u64, u64)> = all_q_primes
        .iter()
        .filter(|&&q| q >= qmin)
        .flat_map(|&q| {
            find_polynomial_roots_mod_p(&f_i64, q)
                .into_iter()
                .map(move |r| (q, r))
        })
        .collect();

    // Parallel sieve across special-q
    let all_hits: Vec<Vec<SieveHit>> = special_qs
        .par_iter()
        .map(|&(q, r_q)| {
            let qlat = reduce_qlattice(q, r_q, 1.0);

            // Precompute transformed roots for each FB prime.
            // Algebraic: R' = (R*b1 - a1) * inv(a0 - R*b0) mod p
            // Rational:  R_rat = (m*b1 - a1) * inv(a0 - m*b0) mod p
            let mut rat_roots: Vec<Option<u64>> = Vec::with_capacity(fb.primes.len());
            let mut alg_roots: Vec<Vec<u64>> = Vec::with_capacity(fb.primes.len());

            for (i, &p) in fb.primes.iter().enumerate() {
                let p_i128 = p as i128;
                let mmp = m_mod_p[i];

                // Rational transformed root
                let rat_denom =
                    ((qlat.a0 as i128 - mmp as i128 * qlat.b0 as i128) % p_i128 + p_i128) % p_i128;
                let rat_numer =
                    ((mmp as i128 * qlat.b1 as i128 - qlat.a1 as i128) % p_i128 + p_i128) % p_i128;
                let rr = if rat_denom == 0 {
                    None
                } else {
                    mod_inverse_u64(rat_denom as u64, p)
                        .map(|inv| ((rat_numer as u128 * inv as u128) % p as u128) as u64)
                };
                rat_roots.push(rr);

                // Algebraic transformed roots
                let mut ar = Vec::with_capacity(fb.algebraic_roots[i].len());
                for &root_r in &fb.algebraic_roots[i] {
                    let denom = ((qlat.a0 as i128 - root_r as i128 * qlat.b0 as i128) % p_i128
                        + p_i128)
                        % p_i128;
                    let numer = ((root_r as i128 * qlat.b1 as i128 - qlat.a1 as i128) % p_i128
                        + p_i128)
                        % p_i128;
                    if denom == 0 {
                        continue;
                    }
                    if let Some(inv) = mod_inverse_u64(denom as u64, p) {
                        ar.push(((numer as u128 * inv as u128) % p as u128) as u64);
                    }
                }
                alg_roots.push(ar);
            }

            // Per-j-line sieve
            let mut local_hits = Vec::new();
            let mut rat_sieve = vec![0u16; sieve_width];
            let mut alg_sieve = vec![0u16; sieve_width];

            for j in 1..=j_max {
                // Clear sieve arrays
                rat_sieve.fill(0);
                alg_sieve.fill(0);

                // Sieve each FB prime
                for (pi, &p) in fb.primes.iter().enumerate() {
                    let log_p = fb.log_p[pi] as u16;
                    let p_usize = p as usize;

                    // Rational side
                    if let Some(r_rat) = rat_roots[pi] {
                        let c = ((j as u128 * r_rat as u128) % p as u128) as u64;
                        let start = ((c + half_width as u64) % p) as usize;
                        let mut idx = start;
                        while idx < sieve_width {
                            rat_sieve[idx] = rat_sieve[idx].saturating_add(log_p);
                            idx += p_usize;
                        }
                    }

                    // Algebraic side
                    for &r_alg in &alg_roots[pi] {
                        let c = ((j as u128 * r_alg as u128) % p as u128) as u64;
                        let start = ((c + half_width as u64) % p) as usize;
                        let mut idx = start;
                        while idx < sieve_width {
                            alg_sieve[idx] = alg_sieve[idx].saturating_add(log_p);
                            idx += p_usize;
                        }
                    }
                }

                // Scan for survivors
                for pos in 0..sieve_width {
                    let i_val = pos as i64 - half_width;
                    if i_val == 0 {
                        continue;
                    }

                    let rat_acc = rat_sieve[pos];
                    let alg_acc = alg_sieve[pos];
                    if rat_acc < 20 || alg_acc < 20 {
                        continue;
                    }

                    // Map (i, j) → (a, b) via q-lattice basis
                    let a_i128 = qlat.a0 as i128 * i_val as i128 + qlat.a1 as i128 * j as i128;
                    let b_i128 = qlat.b0 as i128 * i_val as i128 + qlat.b1 as i128 * j as i128;

                    if b_i128 <= 0 {
                        continue;
                    }

                    // Approximate norm threshold check (f64)
                    let a_f64 = a_i128 as f64;
                    let b_f64 = b_i128 as f64;

                    // Rational: |a - b*m|
                    let rat_norm_f64 = (a_f64 - b_f64 * m_f64).abs();
                    if rat_norm_f64 < 1.0 {
                        continue;
                    }
                    let rat_bits = rat_norm_f64.log2();
                    let rat_threshold = ((rat_bits * 20.0 * 0.5) as u16).saturating_sub(20);
                    if rat_acc < rat_threshold {
                        continue;
                    }

                    // Algebraic: F(a,b) / q (lattice guarantees q | F(a,b))
                    let x = a_f64 / b_f64;
                    let mut alg_val = f_f64[degree];
                    for k in (0..degree).rev() {
                        alg_val = alg_val * x + f_f64[k];
                    }
                    let alg_norm_f64 = (alg_val.abs() * b_f64.powi(degree as i32)) / q as f64;
                    if alg_norm_f64 < 1.0 {
                        continue;
                    }
                    let alg_bits = alg_norm_f64.log2();
                    let alg_threshold = ((alg_bits * 20.0 * 0.5) as u16).saturating_sub(20);
                    if alg_acc < alg_threshold {
                        continue;
                    }

                    // Exact norms via rug::Integer
                    let a_i64 = a_i128 as i64;
                    let b_u64 = b_i128 as u64;

                    let rational_norm = poly.eval_g(a_i64, b_u64);
                    let algebraic_norm = poly.eval_f_homogeneous(a_i64, b_u64);
                    if rational_norm == 0 || algebraic_norm == 0 {
                        continue;
                    }

                    local_hits.push(SieveHit {
                        a: a_i64,
                        b: b_u64,
                        rational_norm,
                        algebraic_norm,
                        special_q: Some((q, r_q)),
                    });
                }
            }

            local_hits
        })
        .collect();

    all_hits.into_iter().flatten().collect()
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
                let val = crate::arith::eval_poly_mod(&[1, 0, 1], r, *p);
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
        assert!(
            !hits.is_empty(),
            "Line sieve should find some candidate pairs"
        );
    }

    #[test]
    fn test_lattice_sieve_finds_hits() {
        use crate::polyselect::select_base_m;

        // 5000101429 = 70001 × 71429 — 33-bit semiprime
        let n = Integer::from(5000101429u64);
        let poly = select_base_m(&n, 3);
        let f_coeffs = poly.f_coeffs();
        let f_i64 = poly_coeffs_to_i64(&f_coeffs).unwrap();
        let fb = build_factor_base(&f_i64, 5_000);

        // Use small sieve region and few special-q for fast test
        let hits = lattice_sieve(&poly, &fb, 8, 6_000, 8_000);
        eprintln!(
            "Lattice sieve test: {} hits from q in [6000, 8000], log_i=8",
            hits.len()
        );
        assert!(
            !hits.is_empty(),
            "Lattice sieve should find candidate pairs"
        );

        // Verify all hits have special_q set and b > 0
        for h in &hits {
            assert!(h.special_q.is_some());
            assert!(h.b > 0);
            let (q, _r) = h.special_q.unwrap();
            assert!(q >= 6_000 && q <= 8_000);
            // Verify q divides algebraic norm
            let alg_abs = h.algebraic_norm.clone().abs();
            assert_eq!(
                Integer::from(&alg_abs % Integer::from(q)),
                0,
                "q={} should divide algebraic norm for (a={}, b={})",
                q,
                h.a,
                h.b
            );
        }
    }
}
