//! Special-q lattice sieve for NFS relation collection.
//!
//! For each special-q prime q with algebraic root r (f(r) ≡ 0 mod q),
//! the sublattice L_q = {(a,b) : a ≡ rb (mod q)} guarantees q | F(a,b).
//! After 2D Gaussian reduction of the basis (q,0),(r,1), we sieve in
//! the reduced (i,j) coordinates where norms are effectively divided by q.

use std::time::Instant;

use classical_nfs::polynomial::NfsPolynomial;
use classical_nfs::sieve::{eval_homogeneous_abs, sieve_primes};
use num_bigint::BigUint;
use num_traits::ToPrimitive;
use rayon::prelude::*;

use crate::sieve_mcmc::{
    compute_roots_fast_pub, fast_log2, trial_divide_u128, FastRootPub, PipelineTimings, Relation,
};
use crate::smoothness::{gcd, trial_divide};

// ─── Configuration ──────────────────────────────────────────────────────────

/// Configuration for the special-q lattice sieve pipeline.
#[derive(Debug, Clone)]
pub struct SpecialQConfig {
    /// Factor base bound (all primes up to this).
    pub fb_bound: u64,
    /// Large prime bound in bits. Cofactors up to 2^lpb accepted as 1LP.
    pub lpb: u32,
    /// Half-width of i-range per special-q (sieve i ∈ [-half_i, half_i)).
    pub sq_half_i: i64,
    /// Max j value per special-q (sieve j ∈ [1, max_j]).
    pub sq_max_j: i64,
    /// Minimum special-q prime (typically fb_bound + 1).
    pub q_min: u64,
    /// Maximum special-q prime.
    pub q_max: u64,
    /// Accept 2-large-prime relations (cofactor ≤ lpb_bound²).
    pub allow_2lp: bool,
}

impl SpecialQConfig {
    pub fn for_bits(bits: u32) -> Self {
        match bits {
            0..=40 => Self {
                fb_bound: 1 << 8,
                lpb: 16,
                sq_half_i: 1 << 9,
                sq_max_j: 1 << 7,
                q_min: (1 << 8) + 1,
                q_max: 1 << 10,
                allow_2lp: true,
            },
            41..=60 => Self {
                fb_bound: 1 << 10,
                lpb: 18,
                sq_half_i: 1 << 10,
                sq_max_j: 1 << 8,
                q_min: (1 << 10) + 1,
                q_max: 1 << 12,
                allow_2lp: true,
            },
            61..=80 => Self {
                fb_bound: 1 << 13,
                lpb: 20,
                sq_half_i: 1 << 11,
                sq_max_j: 1 << 9,
                q_min: (1 << 13) + 1,
                q_max: 1 << 15,
                allow_2lp: true,
            },
            81..=96 => Self {
                fb_bound: 1 << 14,
                lpb: 22,
                sq_half_i: 1 << 12,
                sq_max_j: 1 << 10,
                q_min: (1 << 14) + 1,
                q_max: 1 << 16,
                allow_2lp: true,
            },
            97..=112 => Self {
                fb_bound: 1 << 15,
                lpb: 24,
                sq_half_i: 1 << 12,
                sq_max_j: 1 << 10,
                q_min: (1 << 15) + 1,
                q_max: 1 << 17,
                allow_2lp: true,
            },
            113..=128 => Self {
                fb_bound: 1 << 16,
                lpb: 26,
                sq_half_i: 1 << 12,
                sq_max_j: 1 << 10,
                q_min: (1 << 16) + 1,
                q_max: 1 << 18,
                allow_2lp: true,
            },
            129..=160 => Self {
                fb_bound: 1 << 17,
                lpb: 28,
                sq_half_i: 1 << 13,
                sq_max_j: 1 << 11,
                q_min: (1 << 17) + 1,
                q_max: 1 << 19,
                allow_2lp: true,
            },
            _ => Self {
                fb_bound: 1 << 18,
                lpb: 30,
                sq_half_i: 1 << 14,
                sq_max_j: 1 << 12,
                q_min: (1 << 18) + 1,
                q_max: 1 << 20,
                allow_2lp: true,
            },
        }
    }
}

// ─── Modular arithmetic helpers ─────────────────────────────────────────────

#[inline(always)]
fn mulmod(a: u64, b: u64, m: u64) -> u64 {
    ((a as u128 * b as u128) % m as u128) as u64
}

#[inline(always)]
fn addmod(a: u64, b: u64, m: u64) -> u64 {
    let s = a as u128 + b as u128;
    (s % m as u128) as u64
}

fn modpow_u64(mut base: u64, mut exp: u64, modulus: u64) -> u64 {
    if modulus == 1 {
        return 0;
    }
    let mut result = 1u64;
    base %= modulus;
    while exp > 0 {
        if exp & 1 == 1 {
            result = mulmod(result, base, modulus);
        }
        exp >>= 1;
        if exp > 0 {
            base = mulmod(base, base, modulus);
        }
    }
    result
}

/// Modular inverse via extended Euclidean algorithm. Returns None if gcd(a,m) > 1.
fn mod_inverse_u64(a: u64, m: u64) -> Option<u64> {
    if m <= 1 {
        return None;
    }
    let (mut old_r, mut r) = (a as i128, m as i128);
    let (mut old_s, mut s) = (1i128, 0i128);

    while r != 0 {
        let q = old_r / r;
        let tmp = r;
        r = old_r - q * r;
        old_r = tmp;
        let tmp = s;
        s = old_s - q * s;
        old_s = tmp;
    }

    if old_r != 1 {
        return None;
    }
    Some(((old_s % m as i128 + m as i128) % m as i128) as u64)
}

/// Deterministic Miller-Rabin for all u64 values.
fn is_prime_u64(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    if n == 2 || n == 3 || n == 5 || n == 7 {
        return true;
    }
    if n % 2 == 0 || n % 3 == 0 || n % 5 == 0 {
        return false;
    }

    // Factor n-1 = d * 2^r
    let mut d = n - 1;
    let r = d.trailing_zeros();
    d >>= r;

    // Witnesses sufficient for all u64
    let witnesses: &[u64] = &[2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37];

    'witness: for &a in witnesses {
        if a >= n {
            continue;
        }
        let mut x = modpow_u64(a, d, n);
        if x == 1 || x == n - 1 {
            continue;
        }
        for _ in 0..r - 1 {
            x = mulmod(x, x, n);
            if x == n - 1 {
                continue 'witness;
            }
        }
        return false;
    }
    true
}

/// Pollard rho for factoring u64 composites.
fn pollard_rho_u64(n: u64) -> Option<u64> {
    if n <= 1 {
        return None;
    }
    if n % 2 == 0 {
        return Some(2);
    }

    for c in 1..256u64 {
        let mut x = 2u64;
        let mut y = 2u64;
        let mut d = 1u64;

        while d == 1 {
            x = addmod(mulmod(x, x, n), c, n);
            y = addmod(mulmod(y, y, n), c, n);
            y = addmod(mulmod(y, y, n), c, n);
            d = gcd(x.abs_diff(y), n);
        }

        if d != n {
            return Some(d);
        }
    }
    None
}

// ─── Lattice reduction ──────────────────────────────────────────────────────

/// Reduced 2D lattice basis for a special-q.
#[derive(Debug, Clone)]
struct QLattice {
    q: u64,
    _r: u64,
    a0: i64,
    b0: i64,
    a1: i64,
    b1: i64,
}

/// 2D Gaussian (Lagrange) lattice reduction.
/// Reduces the basis (q, 0), (r, 1) to short vectors.
fn reduce_basis_2d(q: u64, r: u64) -> QLattice {
    let mut a0 = q as i64;
    let mut b0 = 0i64;
    let mut a1 = r as i64;
    let mut b1 = 1i64;

    // Ensure v0 is the shorter vector
    let mut n0 = a0 as i128 * a0 as i128 + b0 as i128 * b0 as i128;
    let mut n1 = a1 as i128 * a1 as i128 + b1 as i128 * b1 as i128;
    if n0 > n1 {
        std::mem::swap(&mut a0, &mut a1);
        std::mem::swap(&mut b0, &mut b1);
        std::mem::swap(&mut n0, &mut n1);
    }

    loop {
        let dot = a0 as i128 * a1 as i128 + b0 as i128 * b1 as i128;
        // Round to nearest integer
        let t = if n0 == 0 {
            break;
        } else {
            let q = dot / n0;
            let r = dot - q * n0;
            if 2 * r.abs() > n0.abs() {
                if (r > 0) == (n0 > 0) {
                    q + 1
                } else {
                    q - 1
                }
            } else {
                q
            }
        };
        if t == 0 {
            break;
        }
        a1 -= t as i64 * a0;
        b1 -= t as i64 * b0;
        n1 = a1 as i128 * a1 as i128 + b1 as i128 * b1 as i128;

        if n1 < n0 {
            std::mem::swap(&mut a0, &mut a1);
            std::mem::swap(&mut b0, &mut b1);
            std::mem::swap(&mut n0, &mut n1);
        }
    }

    QLattice {
        q,
        _r: r,
        a0,
        b0,
        a1,
        b1,
    }
}

// ─── Sieve pattern precomputation ───────────────────────────────────────────

/// Pre-computed sieve pattern for one (prime, root) pair within a QLattice.
#[derive(Clone)]
enum SievePattern {
    /// Regular: i ≡ j*gamma (mod p). For j-row j, offset advances by gamma.
    Regular { gamma: u64, base_offset: u64 },
    /// Column: hits all i, but only for j ≡ 0 (mod p).
    Column,
    /// Global: hits every cell (both alpha and beta are 0 mod p).
    Everywhere,
}

#[derive(Clone)]
struct SQSieveEntry {
    prime: u64,
    log_p: f32,
    pattern: SievePattern,
}

/// Compute sieve entries for all factor-base primes given a reduced QLattice.
/// Returns (rational_entries, algebraic_entries).
fn compute_sq_sieve_entries(
    primes: &[u64],
    log_primes: &[f32],
    alg_roots: &[FastRootPub],
    lattice: &QLattice,
    m: u64,
    sq_half_i: i64,
) -> (Vec<SQSieveEntry>, Vec<SQSieveEntry>) {
    let i_min = -sq_half_i;

    // Rational entries: condition a ≡ m*b (mod p)
    // In lattice coords: i*(a0 - m*b0) + j*(a1 - m*b1) ≡ 0 (mod p)
    let rat_entries: Vec<SQSieveEntry> = primes
        .iter()
        .enumerate()
        .filter(|(_, &p)| p > 1 && p != lattice.q)
        .map(|(idx, &p)| {
            let p128 = p as u128;
            let m128 = m as u128;
            let alpha = ((lattice.a0 as i128 - m128 as i128 * lattice.b0 as i128) % p as i128
                + p as i128) as u128
                % p128;
            let beta = ((lattice.a1 as i128 - m128 as i128 * lattice.b1 as i128) % p as i128
                + p as i128) as u128
                % p128;
            let alpha = alpha as u64;
            let beta = beta as u64;

            let pattern = make_pattern(alpha, beta, p, i_min);
            SQSieveEntry {
                prime: p,
                log_p: log_primes[idx],
                pattern,
            }
        })
        .collect();

    // Algebraic entries: for each root s of f mod p, a ≡ s*b (mod p)
    // In lattice coords: i*(a0 - s*b0) + j*(a1 - s*b1) ≡ 0 (mod p)
    let alg_entries: Vec<SQSieveEntry> = alg_roots
        .iter()
        .filter(|fr| fr.prime != lattice.q)
        .flat_map(|fr| {
            let p = fr.prime;
            let p128 = p as u128;
            fr.roots.iter().map(move |&s| {
                let alpha = ((lattice.a0 as i128 - s as i128 * lattice.b0 as i128) % p as i128
                    + p as i128) as u128
                    % p128;
                let beta = ((lattice.a1 as i128 - s as i128 * lattice.b1 as i128) % p as i128
                    + p as i128) as u128
                    % p128;
                let alpha = alpha as u64;
                let beta = beta as u64;

                let pattern = make_pattern(alpha, beta, p, i_min);
                SQSieveEntry {
                    prime: p,
                    log_p: fr.log_p,
                    pattern,
                }
            })
        })
        .collect();

    (rat_entries, alg_entries)
}

/// Create a sieve pattern from alpha, beta values.
fn make_pattern(alpha: u64, beta: u64, p: u64, i_min: i64) -> SievePattern {
    if alpha == 0 && beta == 0 {
        SievePattern::Everywhere
    } else if alpha == 0 {
        SievePattern::Column
    } else {
        // gamma = (-beta * alpha^{-1}) mod p
        let inv_alpha = mod_inverse_u64(alpha, p).unwrap_or(0);
        let neg_beta = if beta == 0 { 0 } else { p - beta };
        let gamma = mulmod(neg_beta, inv_alpha, p);

        // Base offset for j=0: i ≡ 0 (mod p), so target = 0.
        // First index in array: (0 - i_min) mod p
        let i_min_mod_p = ((i_min % p as i64) + p as i64) as u64 % p;
        let base_offset = if i_min_mod_p == 0 {
            0
        } else {
            p - i_min_mod_p
        };

        SievePattern::Regular {
            gamma,
            base_offset,
        }
    }
}

// ─── Core sieve pipeline ────────────────────────────────────────────────────

/// Shared data for sieving across all special-q values.
struct SQSharedData {
    primes: Vec<u64>,
    log_primes: Vec<f32>,
    alg_roots: Vec<FastRootPub>,
    m: u64,
    m_f64: f64,
    coeffs_f64: Vec<f64>,
    degree: usize,
    lpb_bound: u64,
    lpb_bound_sq: u128,
    lpb_f32: f32,
    allow_2lp: bool,
}

/// Result from sieving one special-q.
struct SQResult {
    relations: Vec<Relation>,
    candidates: usize,
    sieve_ns: u64,
    scan_ns: u64,
    cofactor_ns: u64,
}

/// Main entry point: collect relations using special-q lattice sieve.
pub fn collect_relations_specialq(
    poly: &NfsPolynomial,
    m: u64,
    config: &SpecialQConfig,
) -> (Vec<Relation>, PipelineTimings) {
    let total_start = Instant::now();
    let setup_start = Instant::now();

    // Factor base primes
    let primes = sieve_primes(config.fb_bound);
    let log_primes: Vec<f32> = primes.iter().map(|&p| (p as f32).log2()).collect();
    let alg_roots = compute_roots_fast_pub(&poly.coefficients, &primes);

    // Special-q primes: all primes in [q_min, q_max]
    let sq_primes = sieve_primes(config.q_max);
    let sq_primes: Vec<u64> = sq_primes
        .into_iter()
        .filter(|&q| q >= config.q_min)
        .collect();

    // Find roots of f mod q for each special-q prime
    let sq_pairs: Vec<(u64, u64)> = sq_primes
        .iter()
        .flat_map(|&q| {
            let roots_for_q = compute_roots_fast_pub(&poly.coefficients, &[q]);
            roots_for_q
                .into_iter()
                .flat_map(move |fr| fr.roots.into_iter().map(move |r| (q, r)))
        })
        .collect();

    let coeffs_f64: Vec<f64> = poly
        .coefficients
        .iter()
        .map(|c| c.to_f64().unwrap_or(0.0))
        .collect();

    let lpb_bound = 1u64 << config.lpb;
    let shared = SQSharedData {
        primes,
        log_primes,
        alg_roots,
        m,
        m_f64: m as f64,
        coeffs_f64,
        degree: poly.degree,
        lpb_bound,
        lpb_bound_sq: (lpb_bound as u128) * (lpb_bound as u128),
        lpb_f32: config.lpb as f32,
        allow_2lp: config.allow_2lp,
    };

    let setup_ms = setup_start.elapsed().as_secs_f64() * 1000.0;

    // Process each (q, r) pair in parallel
    let sieve_start = Instant::now();

    let results: Vec<SQResult> = sq_pairs
        .par_iter()
        .map(|&(q, r)| sieve_one_specialq(q, r, config, &shared, poly))
        .collect();

    let wall_ms = sieve_start.elapsed().as_secs_f64() * 1000.0;

    // Aggregate
    let mut all_relations = Vec::new();
    let mut total_candidates = 0usize;
    let mut total_sieve_ns = 0u64;
    let mut total_scan_ns = 0u64;
    let mut total_cofactor_ns = 0u64;

    for r in &results {
        all_relations.extend(r.relations.iter().cloned());
        total_candidates += r.candidates;
        total_sieve_ns += r.sieve_ns;
        total_scan_ns += r.scan_ns;
        total_cofactor_ns += r.cofactor_ns;
    }

    let threads = rayon::current_num_threads() as f64;
    let sieve_ms = total_sieve_ns as f64 / 1_000_000.0 / threads;
    let scan_ms = total_scan_ns as f64 / 1_000_000.0 / threads;
    let cofactor_ms = total_cofactor_ns as f64 / 1_000_000.0 / threads;

    let full_count = all_relations.iter().filter(|r| r.is_full()).count();
    let partial_count = all_relations.len() - full_count;

    let timings = PipelineTimings {
        setup_ms,
        sieve_ms,
        mcmc_ms: scan_ms,
        cofactor_ms,
        total_ms: total_start.elapsed().as_secs_f64() * 1000.0,
        rows_processed: sq_pairs.len(),
        candidates_found: total_candidates,
        relations_found: all_relations.len(),
        full_relations: full_count,
        partial_relations: partial_count,
    };

    eprintln!(
        "  Special-q: {} pairs, {} rels ({} full, {} partial), {:.0}ms wall, {:.0}ms sieve, {:.0}ms scan, {:.0}ms cofact",
        sq_pairs.len(),
        all_relations.len(),
        full_count,
        partial_count,
        wall_ms,
        sieve_ms,
        scan_ms,
        cofactor_ms,
    );

    (all_relations, timings)
}

/// Sieve one (q, r) special-q pair.
fn sieve_one_specialq(
    q: u64,
    r: u64,
    config: &SpecialQConfig,
    shared: &SQSharedData,
    poly: &NfsPolynomial,
) -> SQResult {
    let lattice = reduce_basis_2d(q, r);
    let half_i = config.sq_half_i;
    let max_j = config.sq_max_j;
    let width = (2 * half_i) as usize;

    // Pre-compute sieve entries for this lattice basis
    let (rat_entries, alg_entries) =
        compute_sq_sieve_entries(&shared.primes, &shared.log_primes, &shared.alg_roots, &lattice, shared.m, half_i);

    // Count "everywhere" patterns for threshold adjustment
    let rat_everywhere: f32 = rat_entries
        .iter()
        .filter(|e| matches!(e.pattern, SievePattern::Everywhere))
        .map(|e| e.log_p)
        .sum();
    let alg_everywhere: f32 = alg_entries
        .iter()
        .filter(|e| matches!(e.pattern, SievePattern::Everywhere))
        .map(|e| e.log_p)
        .sum();

    let mut all_candidates: Vec<(i64, i64)> = Vec::new();
    let mut total_sieve_ns = 0u64;
    let mut total_scan_ns = 0u64;

    // Maintain running offsets for Regular pattern entries
    let mut rat_offsets: Vec<u64> = rat_entries
        .iter()
        .map(|e| match &e.pattern {
            SievePattern::Regular { base_offset, .. } => *base_offset,
            _ => 0,
        })
        .collect();
    let mut alg_offsets: Vec<u64> = alg_entries
        .iter()
        .map(|e| match &e.pattern {
            SievePattern::Regular { base_offset, .. } => *base_offset,
            _ => 0,
        })
        .collect();

    for j in 1..=max_j {
        let j_u64 = j as u64;

        // === SIEVE PHASE ===
        let sieve_start = Instant::now();
        let mut rat_scores = vec![rat_everywhere; width];
        let mut alg_scores = vec![alg_everywhere; width];

        // Rational sieve
        for (idx, entry) in rat_entries.iter().enumerate() {
            match &entry.pattern {
                SievePattern::Regular { gamma, .. } => {
                    let p = entry.prime as usize;
                    let mut pos = rat_offsets[idx] as usize;
                    while pos < width {
                        rat_scores[pos] += entry.log_p;
                        pos += p;
                    }
                    // Advance offset for next j
                    let new_off = rat_offsets[idx] + *gamma;
                    rat_offsets[idx] = if new_off >= entry.prime {
                        new_off - entry.prime
                    } else {
                        new_off
                    };
                }
                SievePattern::Column => {
                    if j_u64 % entry.prime == 0 {
                        for s in rat_scores.iter_mut() {
                            *s += entry.log_p;
                        }
                    }
                }
                SievePattern::Everywhere => {} // Already in initial value
            }
        }

        // Algebraic sieve
        for (idx, entry) in alg_entries.iter().enumerate() {
            match &entry.pattern {
                SievePattern::Regular { gamma, .. } => {
                    let p = entry.prime as usize;
                    let mut pos = alg_offsets[idx] as usize;
                    while pos < width {
                        alg_scores[pos] += entry.log_p;
                        pos += p;
                    }
                    let new_off = alg_offsets[idx] + *gamma;
                    alg_offsets[idx] = if new_off >= entry.prime {
                        new_off - entry.prime
                    } else {
                        new_off
                    };
                }
                SievePattern::Column => {
                    if j_u64 % entry.prime == 0 {
                        for s in alg_scores.iter_mut() {
                            *s += entry.log_p;
                        }
                    }
                }
                SievePattern::Everywhere => {}
            }
        }

        total_sieve_ns += sieve_start.elapsed().as_nanos() as u64;

        // === SCAN PHASE: per-cell threshold with lattice-transformed norms ===
        let scan_start = Instant::now();

        let j_f64 = j as f64;
        let a0_f = lattice.a0 as f64;
        let b0_f = lattice.b0 as f64;
        let a1_f = lattice.a1 as f64;
        let b1_f = lattice.b1 as f64;
        let d = shared.degree;
        let lpb_f32 = shared.lpb_f32;
        let _log_q = (q as f32).log2();

        const MIN_SCORE: f32 = 4.0;

        for idx in 0..width {
            if rat_scores[idx] < MIN_SCORE || alg_scores[idx] < MIN_SCORE {
                continue;
            }

            let i = idx as f64 - half_i as f64;

            // (a, b) = i*(a0, b0) + j*(a1, b1)
            let a_f = i * a0_f + j_f64 * a1_f;
            let b_f = i * b0_f + j_f64 * b1_f;

            if b_f.abs() < 0.5 {
                continue; // b must be positive
            }

            // Rational norm: |a - m*b|
            let rat_norm_f64 = (a_f - b_f * shared.m_f64).abs();
            let rat_log = fast_log2(rat_norm_f64);
            let rat_thresh = (rat_log - lpb_f32).max(0.0);
            if rat_scores[idx] < rat_thresh {
                continue;
            }

            // Algebraic norm: |F(a,b)| / q
            // F(a,b) = b^d * f(a/b), then divide by q
            if b_f.abs() < 1e-10 {
                continue;
            }
            let x = a_f / b_f;
            let mut val = shared.coeffs_f64[d];
            for k in (0..d).rev() {
                val = val * x + shared.coeffs_f64[k];
            }
            let alg_norm_f64 = (val.abs() * b_f.abs().powi(d as i32)) / q as f64;
            let alg_log = fast_log2(alg_norm_f64);
            let alg_thresh = (alg_log - lpb_f32).max(0.0);
            if alg_scores[idx] < alg_thresh {
                continue;
            }

            // Convert to integer (a, b)
            let a_int = (i as i64) * lattice.a0 + j * lattice.a1;
            let b_int = (i as i64) * lattice.b0 + j * lattice.b1;

            if b_int <= 0 || a_int == 0 {
                continue;
            }

            if gcd(a_int.unsigned_abs(), b_int as u64) == 1 {
                all_candidates.push((a_int, b_int));
            }
        }

        total_scan_ns += scan_start.elapsed().as_nanos() as u64;
    }

    // === COFACTORIZATION ===
    let cofactor_start = Instant::now();
    let mut relations = Vec::new();

    for &(a, b) in &all_candidates {
        if let Some(rel) = try_cofactorize_sq(a, b, q, shared, poly) {
            relations.push(rel);
        }
    }

    let cofactor_ns = cofactor_start.elapsed().as_nanos() as u64;

    SQResult {
        relations,
        candidates: all_candidates.len(),
        sieve_ns: total_sieve_ns,
        scan_ns: total_scan_ns,
        cofactor_ns,
    }
}

/// Cofactorize a candidate (a, b) from a special-q lattice.
/// The algebraic norm is guaranteed divisible by q. Supports 2LP.
fn try_cofactorize_sq(
    a: i64,
    b: i64,
    q: u64,
    shared: &SQSharedData,
    poly: &NfsPolynomial,
) -> Option<Relation> {
    // Rational norm: |a - b*m|
    let rat_norm_i128 = a as i128 - (b as i128) * (shared.m as i128);
    let rat_norm = rat_norm_i128.unsigned_abs();
    if rat_norm == 0 {
        return None;
    }

    // Trial divide rational side
    let (rat_exps, rat_cofactor) = if rat_norm <= u64::MAX as u128 {
        let (e, c) = trial_divide(rat_norm as u64, &shared.primes);
        (e, c as u128)
    } else {
        trial_divide_u128(rat_norm, &shared.primes)
    };

    // Check rational cofactor
    let rat_cofactor_u64 = rat_cofactor as u64;
    if rat_cofactor > 1 {
        if shared.allow_2lp {
            if rat_cofactor > shared.lpb_bound_sq {
                return None;
            }
        } else if rat_cofactor > shared.lpb_bound as u128 {
            return None;
        }
    }

    // Verify rational 2LP
    if rat_cofactor_u64 > shared.lpb_bound && shared.allow_2lp {
        if !can_split_2lp(rat_cofactor_u64, shared.lpb_bound) {
            return None;
        }
    }

    // Algebraic norm: |F(a,b)| — guaranteed divisible by q
    let alg_norm_big = eval_homogeneous_abs(&poly.coefficients, a, b);

    // Divide by q
    let q_big = BigUint::from(q);
    if &alg_norm_big % &q_big != BigUint::from(0u32) {
        return None; // Shouldn't happen for valid lattice points
    }
    let alg_norm_div_q = &alg_norm_big / &q_big;

    let alg_norm: u128 = match alg_norm_div_q.to_u128() {
        Some(v) if v > 0 => v,
        _ => return None,
    };

    // Trial divide algebraic side
    let (alg_exps, alg_cofactor) = trial_divide_u128(alg_norm, &shared.primes);

    // Check algebraic cofactor
    let alg_cofactor_u64 = alg_cofactor as u64;
    if alg_cofactor > 1 {
        if shared.allow_2lp {
            if alg_cofactor > shared.lpb_bound_sq {
                return None;
            }
        } else if alg_cofactor > shared.lpb_bound as u128 {
            return None;
        }
    }

    // Verify algebraic 2LP
    if alg_cofactor_u64 > shared.lpb_bound && shared.allow_2lp {
        if !can_split_2lp(alg_cofactor_u64, shared.lpb_bound) {
            return None;
        }
    }

    // Build factor lists (including q on the algebraic side)
    let rat_factors: Vec<(u64, u32)> = shared
        .primes
        .iter()
        .zip(rat_exps.iter())
        .filter(|(_, &e)| e > 0)
        .map(|(&p, &e)| (p, e))
        .collect();

    let mut alg_factors: Vec<(u64, u32)> = shared
        .primes
        .iter()
        .zip(alg_exps.iter())
        .filter(|(_, &e)| e > 0)
        .map(|(&p, &e)| (p, e))
        .collect();
    // Add the special-q prime as a known algebraic factor
    alg_factors.push((q, 1));

    Some(Relation {
        a,
        b,
        rat_factors,
        alg_factors,
        rat_cofactor: rat_cofactor_u64,
        alg_cofactor: alg_cofactor_u64,
    })
}

/// Check if a cofactor can be split into two primes, each ≤ lpb_bound.
fn can_split_2lp(cofactor: u64, lpb_bound: u64) -> bool {
    if cofactor <= lpb_bound {
        return true; // 1LP is fine
    }

    // If prime, it's a single large prime > lpb_bound → reject
    if is_prime_u64(cofactor) {
        return false;
    }

    // Try to factor into p1 * p2
    if let Some(p1) = pollard_rho_u64(cofactor) {
        let p2 = cofactor / p1;
        // Both factors must be ≤ lpb_bound
        p1 <= lpb_bound && p2 <= lpb_bound
    } else {
        false
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use classical_nfs::polynomial::select_polynomial;

    #[test]
    fn test_reduce_basis_2d() {
        // q=65537, r=30000
        let lat = reduce_basis_2d(65537, 30000);
        // Verify: det(basis) = a0*b1 - a1*b0 should equal ±q
        let det = (lat.a0 as i128) * (lat.b1 as i128) - (lat.a1 as i128) * (lat.b0 as i128);
        assert_eq!(det.unsigned_abs() as u64, 65537);
        // Both vectors should be short (< 2*sqrt(q))
        let n0 = (lat.a0 as f64).hypot(lat.b0 as f64);
        let n1 = (lat.a1 as f64).hypot(lat.b1 as f64);
        let sqrt_q = (65537.0f64).sqrt();
        assert!(n0 < 2.0 * sqrt_q, "v0 too long: {n0} >= {}", 2.0 * sqrt_q);
        assert!(
            n1 < 2.5 * sqrt_q,
            "v1 too long: {n1} >= {}",
            2.5 * sqrt_q
        );
    }

    #[test]
    fn test_reduce_basis_small() {
        let lat = reduce_basis_2d(17, 5);
        let det = (lat.a0 as i128) * (lat.b1 as i128) - (lat.a1 as i128) * (lat.b0 as i128);
        assert_eq!(det.unsigned_abs() as u64, 17);
    }

    #[test]
    fn test_mod_inverse() {
        assert_eq!(mod_inverse_u64(3, 7), Some(5)); // 3*5 = 15 ≡ 1 mod 7
        assert_eq!(mod_inverse_u64(2, 4), None); // gcd(2,4) = 2
        let inv = mod_inverse_u64(12345, 65537).unwrap();
        assert_eq!(mulmod(12345, inv, 65537), 1);
    }

    #[test]
    fn test_is_prime() {
        assert!(is_prime_u64(2));
        assert!(is_prime_u64(3));
        assert!(!is_prime_u64(4));
        assert!(is_prime_u64(65537));
        assert!(!is_prime_u64(65536));
        assert!(is_prime_u64(1_000_000_007));
    }

    #[test]
    fn test_pollard_rho() {
        let n = 8051u64; // 83 * 97
        let f = pollard_rho_u64(n).unwrap();
        assert!(f == 83 || f == 97);
        assert_eq!(n % f, 0);
    }

    #[test]
    fn test_can_split_2lp() {
        // 83 * 97 = 8051, both ≤ 1000
        assert!(can_split_2lp(8051, 1000));
        // 83 * 97 = 8051, bound = 90 → 97 > 90 → fails
        assert!(!can_split_2lp(8051, 90));
        // Prime > bound → fails
        assert!(!can_split_2lp(65537, 1000));
        // Small prime → 1LP → ok
        assert!(can_split_2lp(97, 1000));
    }

    #[test]
    fn test_specialq_finds_relations_32bit() {
        let n = BigUint::from(2214461131u64);
        let poly = select_polynomial(&n, 3);
        let m = poly.m.to_u64().unwrap();
        let config = SpecialQConfig::for_bits(32);

        let (relations, timings) = collect_relations_specialq(&poly, m, &config);

        eprintln!(
            "SQ 32-bit: {} relations ({} full, {} partial), {} candidates in {:.1}ms",
            relations.len(),
            timings.full_relations,
            timings.partial_relations,
            timings.candidates_found,
            timings.total_ms
        );

        assert!(
            !relations.is_empty(),
            "Special-q should find relations for 32-bit"
        );
    }

    #[test]
    fn test_specialq_finds_relations_64bit() {
        let n = BigUint::from(13763568394002235027u64);
        let poly = select_polynomial(&n, 3);
        let m = poly.m.to_u64().unwrap();
        let config = SpecialQConfig::for_bits(64);

        let (relations, timings) = collect_relations_specialq(&poly, m, &config);

        eprintln!(
            "SQ 64-bit: {} relations ({} full, {} partial), {} candidates in {:.1}ms",
            relations.len(),
            timings.full_relations,
            timings.partial_relations,
            timings.candidates_found,
            timings.total_ms
        );

        assert!(
            !relations.is_empty(),
            "Special-q should find relations for 64-bit"
        );
    }
}
