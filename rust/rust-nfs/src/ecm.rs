//! Standalone ECM (Elliptic Curve Method) for multi-precision integers.
//!
//! Uses `rug::Integer` (GMP-backed) for arbitrary-precision arithmetic,
//! targeting 100-200 bit semiprimes where the u64 cofactor ECM cannot operate.
//!
//! Montgomery curve `By^2 = x^3 + Ax^2 + x` in projective coordinates `(X : Z)`,
//! with Suyama parameterization for 12-torsion starting points.
//!
//! Two phases:
//! - Phase 1: scalar multiply by lcm(1..B1) -- catches factors with B1-smooth group order
//! - Phase 2: baby-step giant-step for primes in (B1, B2] -- catches factors with one
//!   large prime in the group order
//!
//! For c45 (~148-bit balanced semiprimes with ~74-bit factors), ECM with B1=50000
//! has roughly 1/50 success probability per curve. With 200 curves and rayon
//! parallelism, expected wall-clock time is well under 1 second.

use rug::Assign;
use rug::Integer;

/// A point on a Montgomery curve in projective coordinates `(X : Z)`.
///
/// Uses pre-allocated Integers to minimize GMP heap allocations in the hot loop.
#[derive(Clone, Debug)]
struct MontPoint {
    x: Integer,
    z: Integer,
}

/// Scratch space for Montgomery curve operations.
///
/// Pre-allocates all temporary Integers needed by mont_double and mont_add
/// so the hot loop does zero heap allocations. For 148-bit N, each Integer
/// needs ~3 GMP limbs (192 bits); pre-allocating avoids thousands of malloc/free
/// calls per scalar multiplication.
struct MontScratch {
    s1: Integer,
    s2: Integer,
    s3: Integer,
    s4: Integer,
    s5: Integer,
    s6: Integer,
}

impl MontScratch {
    fn new() -> Self {
        MontScratch {
            s1: Integer::new(),
            s2: Integer::new(),
            s3: Integer::new(),
            s4: Integer::new(),
            s5: Integer::new(),
            s6: Integer::new(),
        }
    }
}

/// Factor `n` using ECM with the given parameters.
///
/// Tries up to `max_curves` Suyama-parameterized curves with Phase 1 bound `b1`
/// and Phase 2 bound `b2`. Returns `Some(factor)` on the first non-trivial
/// factor found, or `None` if all curves fail.
///
/// Uses rayon for parallel curve evaluation.
pub fn ecm_factor(n: &Integer, max_curves: usize, b1: u64, b2: u64) -> Option<Integer> {
    // Pre-sieve primes up to b2 once (shared across all curves).
    let primes = sieve_primes_vec(b2);

    // Parallel curve evaluation: first non-trivial factor wins.
    use rayon::prelude::*;
    (0..max_curves)
        .into_par_iter()
        .find_map_any(|idx| {
            let sigma = 6u64 + idx as u64;
            ecm_one_curve(n, b1, b2, sigma, &primes)
        })
}

/// Factor `n` using ECM, single-threaded (no rayon).
///
/// Useful for benchmarking ST performance.
pub fn ecm_factor_st(n: &Integer, max_curves: usize, b1: u64, b2: u64) -> Option<Integer> {
    let primes = sieve_primes_vec(b2);

    for idx in 0..max_curves {
        let sigma = 6u64 + idx as u64;
        if let Some(f) = ecm_one_curve(n, b1, b2, sigma, &primes) {
            return Some(f);
        }
    }
    None
}

/// Try ECM factoring for `n`, with parameters tuned by bit-size.
///
/// B1/B2/max_curves are chosen based on `n.significant_bits()`, but can be
/// overridden via environment variables `RUST_NFS_ECM_B1`, `RUST_NFS_ECM_B2`,
/// and `RUST_NFS_ECM_CURVES`.
///
/// Returns `Some((factor, elapsed_ms))` on success, `None` on failure.
pub fn try_ecm_factor(n: &Integer) -> Option<(Integer, f64)> {
    let bits = n.significant_bits();

    let (default_b1, default_b2, default_curves) = if bits <= 80 {
        (2_000u64, 200_000u64, 25usize)
    } else if bits <= 100 {
        (5_000, 500_000, 50)
    } else if bits <= 120 {
        (11_000, 1_100_000, 90)
    } else if bits <= 140 {
        (25_000, 2_500_000, 150)
    } else if bits <= 160 {
        (50_000, 5_000_000, 200)
    } else if bits <= 180 {
        (250_000, 25_000_000, 400)
    } else {
        (1_000_000, 100_000_000, 800)
    };

    let b1: u64 = std::env::var("RUST_NFS_ECM_B1")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default_b1);
    let b2: u64 = std::env::var("RUST_NFS_ECM_B2")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default_b2);
    let max_curves: usize = std::env::var("RUST_NFS_ECM_CURVES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default_curves);

    let start = std::time::Instant::now();
    let result = ecm_factor(n, max_curves, b1, b2);
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

    if let Some(f) = result {
        eprintln!(
            "  ecm: factor found in {:.1}ms (B1={}, B2={}, curves={}, bits={}): {}",
            elapsed_ms, b1, b2, max_curves, bits, f
        );
        Some((f, elapsed_ms))
    } else {
        eprintln!(
            "  ecm: no factor after {:.1}ms (B1={}, B2={}, curves={}, bits={})",
            elapsed_ms, b1, b2, max_curves, bits
        );
        None
    }
}

/// Run one ECM curve on `n` with Suyama parameterization from `sigma`.
///
/// Returns `Some(factor)` if a non-trivial factor is found in Phase 1 or Phase 2.
fn ecm_one_curve(n: &Integer, b1: u64, b2: u64, sigma: u64, primes: &[u64]) -> Option<Integer> {
    // Build Montgomery curve via Suyama parameterization.
    // u = sigma^2 - 5 mod n, v = 4*sigma mod n
    let sigma_int = Integer::from(sigma);
    let mut u = Integer::from(&sigma_int * &sigma_int);
    u -= 5u32;
    u.modulo_mut(n);
    let mut v = Integer::from(&sigma_int * 4u32);
    v.modulo_mut(n);

    if u == 0 || v == 0 {
        return None;
    }

    // Starting point Q = (u^3 : v^3)
    let mut u3 = Integer::from(&u * &u);
    u3 *= &u;
    u3.modulo_mut(n);
    let mut v3 = Integer::from(&v * &v);
    v3 *= &v;
    v3.modulo_mut(n);

    // a24 = (A+2)/4 = (v-u)^3 * (3u+v) / (16 * u^3 * v) mod n
    let v_minus_u = mod_sub(&v, &u, n);
    let mut vmu3 = Integer::from(&v_minus_u * &v_minus_u);
    vmu3 *= &v_minus_u;
    vmu3.modulo_mut(n);
    let mut three_u_plus_v = Integer::from(&u * 3u32);
    three_u_plus_v += &v;
    three_u_plus_v.modulo_mut(n);
    let mut numerator = Integer::from(&vmu3 * &three_u_plus_v);
    numerator.modulo_mut(n);

    let mut denom = Integer::from(&u3 * &v);
    denom *= 16u32;
    denom.modulo_mut(n);

    // Compute modular inverse of denom; if gcd(denom, n) > 1, we found a factor.
    let denom_inv = match mod_inverse(&denom, n) {
        Ok(inv) => inv,
        Err(g) => {
            if g > 1u32 && g < *n {
                return Some(g);
            }
            return None;
        }
    };

    let mut a24 = Integer::from(&numerator * &denom_inv);
    a24.modulo_mut(n);

    let mut q = MontPoint {
        x: u3.clone(),
        z: v3.clone(),
    };

    let mut scratch = MontScratch::new();

    // ---------------------------------------------------------------
    // Phase 1: multiply Q by lcm(1..B1)
    // ---------------------------------------------------------------
    let mut accum = Integer::from(1);
    let mut step_count = 0u32;

    for &p in primes.iter() {
        if p > b1 {
            break;
        }
        let mut pk = p;
        while pk <= b1 {
            mont_ladder_inplace(&mut q, p, &a24, n, &mut scratch);
            pk = pk.saturating_mul(p);
        }

        // Accumulate Z into product for batched GCD
        accum *= &q.z;
        accum.modulo_mut(n);
        step_count += 1;

        if step_count % 32 == 0 {
            if let Some(f) = check_gcd(&accum, n) {
                return Some(f);
            }
            if accum == 0 {
                return ecm_one_curve_careful(n, b1, b2, sigma, primes);
            }
        }
    }

    // Final Phase 1 GCD check
    if let Some(f) = check_gcd(&accum, n) {
        return Some(f);
    }
    if accum == 0 {
        return ecm_one_curve_careful(n, b1, b2, sigma, primes);
    }

    // ---------------------------------------------------------------
    // Phase 2: standard continuation with baby-step giant-step
    // ---------------------------------------------------------------
    let mut accum2 = Integer::from(1);
    let mut batch_count = 0u32;
    const PHASE2_BATCH: u32 = 64;

    let q_base = q.clone();

    // Baby step size D ~= sqrt(B2 - B1)
    let d_step = ((b2 - b1) as f64).sqrt() as u64;
    let d = if d_step < 2 { 2 } else { d_step | 1 };

    // Precompute baby steps: [1]Q, [2]Q, ..., [d]Q
    let baby = precompute_baby_steps(&q_base, d, &a24, n, &mut scratch);

    let d_q = &baby[baby.len() - 1]; // [d]Q

    // Giant step: start at B1 rounded up to next multiple of d
    let first_giant = ((b1 / d) + 1) * d;
    let mut giant_q = mont_ladder_new(&q_base, first_giant, &a24, n, &mut scratch);
    let mut prev_giant_q = if first_giant >= d {
        mont_ladder_new(&q_base, first_giant - d, &a24, n, &mut scratch)
    } else {
        MontPoint {
            x: Integer::from(1),
            z: Integer::from(0),
        }
    };

    let mut cross = Integer::new();
    let mut tmp = Integer::new();

    let mut giant_val = first_giant;
    while giant_val <= b2 + d {
        for b_idx in 0..baby.len() {
            let b_offset = (b_idx + 1) as u64;
            let val_plus = giant_val + b_offset;
            let val_minus = if giant_val >= b_offset {
                giant_val - b_offset
            } else {
                0
            };

            if val_plus > b1 && val_plus <= b2 && is_in_prime_list(val_plus, primes) {
                // cross = (X_giant * Z_baby - Z_giant * X_baby) mod n
                cross.assign(&giant_q.x * &baby[b_idx].z);
                tmp.assign(&giant_q.z * &baby[b_idx].x);
                cross -= &tmp;
                cross.modulo_mut(n);
                accum2 *= &cross;
                accum2.modulo_mut(n);
                batch_count += 1;
            }

            if val_minus > b1 && val_minus <= b2 && is_in_prime_list(val_minus, primes) {
                cross.assign(&giant_q.x * &baby[b_idx].z);
                tmp.assign(&giant_q.z * &baby[b_idx].x);
                cross -= &tmp;
                cross.modulo_mut(n);
                accum2 *= &cross;
                accum2.modulo_mut(n);
                batch_count += 1;
            }

            if batch_count >= PHASE2_BATCH {
                if let Some(f) = check_gcd(&accum2, n) {
                    return Some(f);
                }
                batch_count = 0;
            }
        }

        // Advance giant step
        let new_giant = mont_add_new(&giant_q, d_q, &prev_giant_q, n, &mut scratch);
        prev_giant_q = giant_q;
        giant_q = new_giant;
        giant_val += d;
    }

    check_gcd(&accum2, n)
}

/// Careful per-prime GCD fallback when batched Phase 1 overshoots.
fn ecm_one_curve_careful(
    n: &Integer,
    b1: u64,
    _b2: u64,
    sigma: u64,
    primes: &[u64],
) -> Option<Integer> {
    let sigma_int = Integer::from(sigma);
    let mut u = Integer::from(&sigma_int * &sigma_int);
    u -= 5u32;
    u.modulo_mut(n);
    let mut v = Integer::from(&sigma_int * 4u32);
    v.modulo_mut(n);
    if u == 0 || v == 0 {
        return None;
    }

    let mut u3 = Integer::from(&u * &u);
    u3 *= &u;
    u3.modulo_mut(n);
    let mut v3 = Integer::from(&v * &v);
    v3 *= &v;
    v3.modulo_mut(n);

    let v_minus_u = mod_sub(&v, &u, n);
    let mut vmu3 = Integer::from(&v_minus_u * &v_minus_u);
    vmu3 *= &v_minus_u;
    vmu3.modulo_mut(n);
    let mut three_u_plus_v = Integer::from(&u * 3u32);
    three_u_plus_v += &v;
    three_u_plus_v.modulo_mut(n);
    let mut numerator = Integer::from(&vmu3 * &three_u_plus_v);
    numerator.modulo_mut(n);
    let mut denom = Integer::from(&u3 * &v);
    denom *= 16u32;
    denom.modulo_mut(n);

    let denom_inv = match mod_inverse(&denom, n) {
        Ok(inv) => inv,
        Err(g) => {
            if g > 1u32 && g < *n {
                return Some(g);
            }
            return None;
        }
    };

    let mut a24 = Integer::from(&numerator * &denom_inv);
    a24.modulo_mut(n);
    let mut q = MontPoint {
        x: u3.clone(),
        z: v3.clone(),
    };
    let mut scratch = MontScratch::new();

    for &p in primes.iter() {
        if p > b1 {
            break;
        }
        let mut pk = p;
        while pk <= b1 {
            mont_ladder_inplace(&mut q, p, &a24, n, &mut scratch);
            pk = pk.saturating_mul(p);
        }
        let g = Integer::from(n.gcd_ref(&q.z));
        if g > 1u32 && g < *n {
            return Some(g);
        }
        if g == *n {
            return None;
        }
    }

    None
}

// ---------------------------------------------------------------------------
// Montgomery curve point operations -- allocation-optimized
// ---------------------------------------------------------------------------

/// Montgomery double: `result = [2]p`, using scratch space for zero allocation.
///
/// Formulas (all mod n):
/// ```text
/// sum  = (X + Z)
/// diff = (X - Z)
/// u    = sum^2
/// v    = diff^2
/// w    = u - v
/// X2   = u * v mod n
/// Z2   = w * (v + a24 * w) mod n
/// ```
#[inline]
fn mont_double_into(result: &mut MontPoint, p: &MontPoint, a24: &Integer, n: &Integer, sc: &mut MontScratch) {
    // s1 = X + Z
    sc.s1.assign(&p.x + &p.z);
    // s2 = X - Z (mod n, keep positive)
    sc.s2.assign(&p.x - &p.z);
    if sc.s2 < 0 {
        sc.s2 += n;
    }
    // s3 = u = s1^2 mod n
    sc.s3.assign(&sc.s1 * &sc.s1);
    sc.s3.modulo_mut(n);
    // s4 = v = s2^2 mod n
    sc.s4.assign(&sc.s2 * &sc.s2);
    sc.s4.modulo_mut(n);
    // X2 = u * v mod n
    result.x.assign(&sc.s3 * &sc.s4);
    result.x.modulo_mut(n);
    // s5 = w = u - v
    sc.s5.assign(&sc.s3 - &sc.s4);
    if sc.s5 < 0 {
        sc.s5 += n;
    }
    // s6 = a24 * w mod n
    sc.s6.assign(a24 * &sc.s5);
    sc.s6.modulo_mut(n);
    // s6 = v + a24*w
    sc.s6 += &sc.s4;
    // Z2 = w * (v + a24*w) mod n
    result.z.assign(&sc.s5 * &sc.s6);
    result.z.modulo_mut(n);
}

/// Montgomery differential addition: `result = p + q`, given `diff = p - q`.
///
/// ```text
/// u1 = (Xp - Zp) * (Xq + Zq)
/// u2 = (Xp + Zp) * (Xq - Zq)
/// X3 = Zdiff * (u1 + u2)^2 mod n
/// Z3 = Xdiff * (u1 - u2)^2 mod n
/// ```
#[inline]
fn mont_add_into(
    result: &mut MontPoint,
    p: &MontPoint,
    q: &MontPoint,
    diff: &MontPoint,
    n: &Integer,
    sc: &mut MontScratch,
) {
    // s1 = Xp - Zp
    sc.s1.assign(&p.x - &p.z);
    if sc.s1 < 0 {
        sc.s1 += n;
    }
    // s2 = Xq + Zq
    sc.s2.assign(&q.x + &q.z);
    // s3 = u1 = s1 * s2 mod n
    sc.s3.assign(&sc.s1 * &sc.s2);
    sc.s3.modulo_mut(n);

    // s1 = Xp + Zp
    sc.s1.assign(&p.x + &p.z);
    // s2 = Xq - Zq
    sc.s2.assign(&q.x - &q.z);
    if sc.s2 < 0 {
        sc.s2 += n;
    }
    // s4 = u2 = s1 * s2 mod n
    sc.s4.assign(&sc.s1 * &sc.s2);
    sc.s4.modulo_mut(n);

    // s1 = u1 + u2
    sc.s1.assign(&sc.s3 + &sc.s4);
    // s2 = (u1 + u2)^2 mod n
    sc.s2.assign(&sc.s1 * &sc.s1);
    sc.s2.modulo_mut(n);
    // X3 = Zdiff * (u1+u2)^2 mod n
    result.x.assign(&diff.z * &sc.s2);
    result.x.modulo_mut(n);

    // s1 = u1 - u2
    sc.s1.assign(&sc.s3 - &sc.s4);
    if sc.s1 < 0 {
        sc.s1 += n;
    }
    // s2 = (u1-u2)^2 mod n
    sc.s2.assign(&sc.s1 * &sc.s1);
    sc.s2.modulo_mut(n);
    // Z3 = Xdiff * (u1-u2)^2 mod n
    result.z.assign(&diff.x * &sc.s2);
    result.z.modulo_mut(n);
}

/// Montgomery ladder: compute `[k]P` in-place (mutates `p`).
fn mont_ladder_inplace(p: &mut MontPoint, k: u64, a24: &Integer, n: &Integer, sc: &mut MontScratch) {
    if k == 0 {
        p.x.assign(1);
        p.z.assign(0);
        return;
    }
    if k == 1 {
        return;
    }

    // We need the original P for differential addition.
    let p_orig = p.clone();

    // r0 = P, r1 = [2]P
    let mut r0 = p_orig.clone();
    let mut r1 = MontPoint {
        x: Integer::new(),
        z: Integer::new(),
    };
    mont_double_into(&mut r1, &p_orig, a24, n, sc);

    let mut tmp = MontPoint {
        x: Integer::new(),
        z: Integer::new(),
    };

    let bits = 64 - k.leading_zeros();
    for i in (0..bits - 1).rev() {
        if (k >> i) & 1 == 1 {
            // r0 = r0 + r1 (diff = P), r1 = 2*r1
            mont_add_into(&mut tmp, &r0, &r1, &p_orig, n, sc);
            std::mem::swap(&mut r0, &mut tmp);
            mont_double_into(&mut tmp, &r1, a24, n, sc);
            std::mem::swap(&mut r1, &mut tmp);
        } else {
            // r1 = r0 + r1 (diff = P), r0 = 2*r0
            mont_add_into(&mut tmp, &r0, &r1, &p_orig, n, sc);
            std::mem::swap(&mut r1, &mut tmp);
            mont_double_into(&mut tmp, &r0, a24, n, sc);
            std::mem::swap(&mut r0, &mut tmp);
        }
    }

    p.x.assign(&r0.x);
    p.z.assign(&r0.z);
}

/// Montgomery ladder: return new `[k]P` (does not mutate input).
fn mont_ladder_new(
    p: &MontPoint,
    k: u64,
    a24: &Integer,
    n: &Integer,
    sc: &mut MontScratch,
) -> MontPoint {
    let mut result = p.clone();
    mont_ladder_inplace(&mut result, k, a24, n, sc);
    result
}

/// Allocating version of mont_add for phase 2 step.
fn mont_add_new(
    p: &MontPoint,
    q: &MontPoint,
    diff: &MontPoint,
    n: &Integer,
    sc: &mut MontScratch,
) -> MontPoint {
    let mut result = MontPoint {
        x: Integer::new(),
        z: Integer::new(),
    };
    mont_add_into(&mut result, p, q, diff, n, sc);
    result
}

/// Precompute baby step table: [1]P, [2]P, ..., [d]P.
fn precompute_baby_steps(
    p: &MontPoint,
    d: u64,
    a24: &Integer,
    n: &Integer,
    sc: &mut MontScratch,
) -> Vec<MontPoint> {
    let mut table = Vec::with_capacity(d as usize);
    if d == 0 {
        return table;
    }

    table.push(p.clone());
    if d == 1 {
        return table;
    }

    let mut p2 = MontPoint {
        x: Integer::new(),
        z: Integer::new(),
    };
    mont_double_into(&mut p2, p, a24, n, sc);
    table.push(p2);
    if d == 2 {
        return table;
    }

    for _ in 3..=d {
        let len = table.len();
        let next = mont_add_new(&table[len - 1], &table[0], &table[len - 2], n, sc);
        table.push(next);
    }

    table
}

// ---------------------------------------------------------------------------
// Non-hot-path wrappers for tests and simple callers
// ---------------------------------------------------------------------------

/// Allocating mont_double for use in tests.
#[cfg(test)]
fn mont_double(p: &MontPoint, a24: &Integer, n: &Integer) -> MontPoint {
    let mut result = MontPoint {
        x: Integer::new(),
        z: Integer::new(),
    };
    let mut sc = MontScratch::new();
    mont_double_into(&mut result, p, a24, n, &mut sc);
    result
}

/// Allocating mont_ladder for use in tests.
#[cfg(test)]
fn mont_ladder(p: &MontPoint, k: u64, a24: &Integer, n: &Integer) -> MontPoint {
    let mut sc = MontScratch::new();
    mont_ladder_new(p, k, a24, n, &mut sc)
}

// ---------------------------------------------------------------------------
// Arithmetic helpers
// ---------------------------------------------------------------------------

/// Modular subtraction: `(a - b) mod n`, always non-negative.
#[inline]
fn mod_sub(a: &Integer, b: &Integer, n: &Integer) -> Integer {
    let mut diff = Integer::from(a - b);
    if diff < 0 {
        diff += n;
    }
    diff.modulo_mut(n);
    diff
}

/// Modular inverse: returns `Ok(a^{-1} mod n)` if gcd(a, n) == 1,
/// or `Err(gcd(a, n))` if a non-trivial GCD is found.
fn mod_inverse(a: &Integer, n: &Integer) -> Result<Integer, Integer> {
    let g = Integer::from(a.gcd_ref(n));
    if g == 1u32 {
        match a.clone().invert(n) {
            Ok(inv) => Ok(inv),
            Err(_) => Err(g),
        }
    } else {
        Err(g)
    }
}

/// Check if `gcd(accum, n)` yields a non-trivial factor.
#[inline]
fn check_gcd(accum: &Integer, n: &Integer) -> Option<Integer> {
    if *accum == 0u32 {
        return None;
    }
    let g = Integer::from(accum.gcd_ref(n));
    if g > 1u32 && g < *n {
        Some(g)
    } else {
        None
    }
}

/// Check whether `val` is in the sorted prime list via binary search.
#[inline]
fn is_in_prime_list(val: u64, primes: &[u64]) -> bool {
    primes.binary_search(&val).is_ok()
}

/// Sieve of Eratosthenes up to `bound`, returning a sorted Vec<u64>.
fn sieve_primes_vec(bound: u64) -> Vec<u64> {
    if bound < 2 {
        return Vec::new();
    }
    let n = bound as usize;
    let mut is_prime = vec![true; n + 1];
    is_prime[0] = false;
    is_prime[1] = false;

    let mut i = 2;
    while i * i <= n {
        if is_prime[i] {
            let mut j = i * i;
            while j <= n {
                is_prime[j] = false;
                j += i;
            }
        }
        i += 1;
    }

    is_prime
        .iter()
        .enumerate()
        .filter_map(|(idx, &flag)| if flag { Some(idx as u64) } else { None })
        .collect()
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to create a semiprime from two primes.
    fn semiprime(p: &str, q: &str) -> Integer {
        let pi: Integer = p.parse().unwrap();
        let qi: Integer = q.parse().unwrap();
        Integer::from(&pi * &qi)
    }

    #[test]
    fn test_mont_double_basic() {
        let n = Integer::from(1009u64 * 1013);
        let a24 = Integer::from(3u32);
        let p = MontPoint {
            x: Integer::from(7u32),
            z: Integer::from(1u32),
        };
        let p2 = mont_double(&p, &a24, &n);
        assert!(p2.x < n);
        assert!(p2.z < n);
    }

    #[test]
    fn test_mont_ladder_identity() {
        let n = Integer::from(1009u64 * 1013);
        let a24 = Integer::from(3u32);
        let p = MontPoint {
            x: Integer::from(7u32),
            z: Integer::from(1u32),
        };
        let p1 = mont_ladder(&p, 1, &a24, &n);
        assert_eq!(p1.x, p.x);
        assert_eq!(p1.z, p.z);
    }

    #[test]
    fn test_ecm_small_semiprime() {
        let n = Integer::from(1009u64 * 1013);
        let result = ecm_factor_st(&n, 50, 500, 5000);
        assert!(result.is_some(), "ECM should factor 1009*1013");
        let f = result.unwrap();
        assert!(Integer::from(&n % &f) == 0u32, "factor should divide n");
        assert!(f > 1u32 && f < n, "factor should be non-trivial");
    }

    #[test]
    fn test_ecm_medium_semiprime() {
        let n = Integer::from(10007u64 * 10009);
        let result = ecm_factor_st(&n, 50, 1000, 10000);
        assert!(result.is_some(), "ECM should factor 10007*10009");
        let f = result.unwrap();
        assert!(Integer::from(&n % &f) == 0u32);
        assert!(f > 1u32 && f < n);
    }

    #[test]
    fn test_ecm_32bit_semiprime() {
        let n = Integer::from(65521u64 * 65537);
        let result = ecm_factor_st(&n, 100, 2000, 50000);
        assert!(result.is_some(), "ECM should factor 65521*65537");
        let f = result.unwrap();
        assert!(Integer::from(&n % &f) == 0u32);
        assert!(f > 1u32 && f < n);
    }

    #[test]
    fn test_ecm_64bit_semiprime() {
        let p: u64 = 2_147_483_659;
        let q: u64 = 2_147_483_693;
        let n = Integer::from(p) * Integer::from(q);
        let result = ecm_factor_st(&n, 200, 5000, 500_000);
        assert!(result.is_some(), "ECM should factor 64-bit semiprime");
        let f = result.unwrap();
        assert!(Integer::from(&n % &f) == 0u32);
        assert!(f > 1u32 && f < n);
    }

    #[test]
    fn test_ecm_prime_returns_none() {
        let n: Integer = "104729".parse().unwrap();
        let result = ecm_factor_st(&n, 20, 500, 5000);
        assert!(result.is_none(), "ECM on a prime should return None");
    }

    #[test]
    fn test_ecm_parallel_small() {
        let n = Integer::from(1009u64 * 1013);
        let result = ecm_factor(&n, 50, 500, 5000);
        assert!(result.is_some(), "Parallel ECM should factor 1009*1013");
        let f = result.unwrap();
        assert!(Integer::from(&n % &f) == 0u32);
    }

    #[test]
    fn test_ecm_100bit_semiprime() {
        let n = semiprime("1125899906842679", "1125899906842747");
        assert!(n.significant_bits() >= 99);
        let result = ecm_factor_st(&n, 200, 10000, 1_000_000);
        assert!(result.is_some(), "ECM should factor 100-bit semiprime");
        let f = result.unwrap();
        assert!(Integer::from(&n % &f) == 0u32);
    }

    #[test]
    fn test_ecm_120bit_semiprime() {
        let n = semiprime("1152921504606847009", "1152921504606847087");
        assert!(n.significant_bits() >= 119);
        let result = ecm_factor_st(&n, 200, 20000, 2_000_000);
        assert!(result.is_some(), "ECM should factor 120-bit semiprime");
        let f = result.unwrap();
        assert!(Integer::from(&n % &f) == 0u32);
    }

    #[test]
    fn test_mod_inverse_invertible() {
        let n = Integer::from(1009u64 * 1013);
        let a = Integer::from(7u32);
        let inv = mod_inverse(&a, &n).unwrap();
        let product = Integer::from(&a * &inv).modulo(&n);
        assert_eq!(product, 1u32, "a * a^-1 should be 1 mod n");
    }

    #[test]
    fn test_mod_inverse_non_invertible() {
        let n = Integer::from(1009u64 * 1013);
        let a = Integer::from(1009u32);
        match mod_inverse(&a, &n) {
            Err(g) => {
                assert!(g > 1u32 && g <= 1009u32);
            }
            Ok(_) => panic!("1009 should not be invertible mod 1009*1013"),
        }
    }

    #[test]
    fn test_sieve_primes_vec() {
        let primes = sieve_primes_vec(30);
        assert_eq!(primes, vec![2, 3, 5, 7, 11, 13, 17, 19, 23, 29]);
    }

    #[test]
    fn test_mod_sub_no_underflow() {
        let n = Integer::from(1009u64);
        let a = Integer::from(3u32);
        let b = Integer::from(1005u32);
        let result = mod_sub(&a, &b, &n);
        assert_eq!(result, 7u32);
    }

    #[test]
    fn test_ecm_148bit_c45_semiprime() {
        // Third c45 test case:
        // p = 10356651620313423478747 (74-bit), q = 9737786813221482516737 (74-bit)
        let n = semiprime("10356651620313423478747", "9737786813221482516737");
        let bits = n.significant_bits();
        eprintln!("c45 case 3: {} bits, {} digits", bits, n.to_string().len());
        assert!(bits >= 136, "should be >= 136 bits, got {}", bits);
        let result = ecm_factor_st(&n, 400, 50000, 5_000_000);
        assert!(result.is_some(), "ECM should factor c45 semiprime");
        let f = result.unwrap();
        assert!(Integer::from(&n % &f) == 0u32);
        assert!(f > 1u32 && f < n);
    }

    #[test]
    fn test_ecm_c45_balanced_semiprime() {
        // Balanced c45: two 74-bit primes (verified with sympy)
        // p = 9502126893814776953359, q = 10021655427541319958881
        // n has 147 bits, 44 digits
        let p: Integer = "9502126893814776953359".parse().unwrap();
        let q: Integer = "10021655427541319958881".parse().unwrap();
        assert!(p.is_probably_prime(30) != rug::integer::IsPrime::No, "p should be prime");
        assert!(q.is_probably_prime(30) != rug::integer::IsPrime::No, "q should be prime");
        let n = Integer::from(&p * &q);
        let bits = n.significant_bits();
        eprintln!("c45 test: n has {} bits, {} digits", bits, n.to_string().len());
        assert!(bits >= 136 && bits <= 150, "expected ~147 bits, got {}", bits);

        let start = std::time::Instant::now();
        let result = ecm_factor_st(&n, 400, 50000, 5_000_000);
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        eprintln!("c45 ST ECM: {:.1}ms", elapsed);

        assert!(result.is_some(), "ECM should factor balanced c45 (147-bit) semiprime");
        let f = result.unwrap();
        assert!(Integer::from(&n % &f) == 0u32);
        assert!(f > 1u32 && f < n);
    }

    #[test]
    fn test_ecm_c45_parallel() {
        let n = semiprime("9502126893814776953359", "10021655427541319958881");
        let start = std::time::Instant::now();
        let result = ecm_factor(&n, 400, 50000, 5_000_000);
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        eprintln!("c45 MT ECM: {:.1}ms", elapsed);

        assert!(result.is_some(), "Parallel ECM should factor balanced c45 semiprime");
        let f = result.unwrap();
        assert!(Integer::from(&n % &f) == 0u32);
    }

    #[test]
    fn test_ecm_c45_second_case() {
        // p = 9652584922799531340689 (74-bit), q = 10079434548832701666223 (74-bit)
        let n = semiprime("9652584922799531340689", "10079434548832701666223");
        let bits = n.significant_bits();
        eprintln!("c45 case 2: {} bits, {} digits", bits, n.to_string().len());

        let start = std::time::Instant::now();
        let result = ecm_factor(&n, 400, 50000, 5_000_000);
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        eprintln!("c45 case 2 MT ECM: {:.1}ms", elapsed);

        assert!(result.is_some(), "ECM should factor second c45 case");
        let f = result.unwrap();
        assert!(Integer::from(&n % &f) == 0u32);
    }

    #[test]
    fn test_try_ecm_factor_auto_params() {
        let n = Integer::from(1009u64 * 1013);
        let result = try_ecm_factor(&n);
        assert!(result.is_some());
        let (f, ms) = result.unwrap();
        assert!(Integer::from(&n % &f) == 0u32);
        eprintln!("try_ecm_factor: found {} in {:.1}ms", f, ms);
    }
}
