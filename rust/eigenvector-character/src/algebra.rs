//! Multiplicative characters of (ℤ/ℓℤ)* for prime ℓ.
//!
//! For a prime ℓ, (ℤ/ℓℤ)* is cyclic of order ℓ−1.  Fix a primitive root g.
//! The Dirichlet character χ_r (r = 0 … ℓ−2) is defined by
//! `chi_r(a) = exp(2*pi*i * r * ind_g(a) / (ell-1))`,
//! where ind_g(a) is the discrete logarithm base g of a in (ℤ/ℓℤ)*.
//! χ_0 is the trivial character (identically 1).
//! For real characters: χ_0 and χ_{(ℓ−1)/2} (Legendre symbol, when 2 | ℓ−1).
//!
//! Key identity used in E21:
//!   g(p) = (1 + p^{k−1}) mod ℓ
//!   g(p) · g(q) = (1 + p^{k−1})(1 + q^{k−1}) = σ_{k−1}(pq) mod ℓ
//! so χ_r(g(p)) · χ_r(g(q)) = χ_r(σ_{k−1}(N) mod ℓ) for all r.

use eisenstein_hunt::arith::mod_pow;

// ---------------------------------------------------------------------------
// Primitive root
// ---------------------------------------------------------------------------

/// Find a primitive root of the prime ℓ by brute force.
///
/// Tests g = 2, 3, … until g has order ℓ−1 in (ℤ/ℓℤ)*.
/// Feasible for ℓ ≤ 50 000 (typical smallest primitive root is very small).
pub fn primitive_root(ell: u64) -> u64 {
    assert!(ell >= 2, "ell must be prime");
    let order = ell - 1;
    let factors = distinct_prime_factors(order);
    'outer: for g in 2..ell {
        // g is a primitive root iff g^(order/p) ≢ 1 (mod ell) for every prime p | order.
        for &p in &factors {
            if mod_pow(g, order / p, ell) == 1 {
                continue 'outer;
            }
        }
        return g;
    }
    panic!("No primitive root found for ell={ell}");
}

/// Distinct prime factors of n (trial division).
fn distinct_prime_factors(mut n: u64) -> Vec<u64> {
    let mut factors = Vec::new();
    let mut d = 2u64;
    while d * d <= n {
        if n % d == 0 {
            factors.push(d);
            while n % d == 0 {
                n /= d;
            }
        }
        d += 1;
    }
    if n > 1 {
        factors.push(n);
    }
    factors
}

// ---------------------------------------------------------------------------
// Discrete-log table
// ---------------------------------------------------------------------------

/// Build a discrete-log table for (ℤ/ℓℤ)* with generator g.
///
/// Returns `Vec<u32>` of length `ell` indexed by a = 0 … ell−1:
/// - `table[0]      = u32::MAX`  (sentinel: 0 ∉ (ℤ/ℓℤ)*)
/// - `table[a]      = k`         such that g^k ≡ a (mod ell)
pub fn build_dlog_table(ell: u64, g: u64) -> Vec<u32> {
    let order = (ell - 1) as usize;
    let mut table = vec![u32::MAX; ell as usize];
    let mut val = 1u64;
    for k in 0..order {
        debug_assert!(
            table[val as usize] == u32::MAX,
            "g={g} is not a primitive root of ell={ell}"
        );
        table[val as usize] = k as u32;
        val = val * g % ell;
    }
    debug_assert_eq!(val, 1, "power cycle did not return to 1 — g={g} not a primitive root of ell={ell}");
    table
}

// ---------------------------------------------------------------------------
// Character evaluation
// ---------------------------------------------------------------------------

/// Real part of χ_r(a): cos(2π · r · dlog_a / order).
///
/// `dlog_a` must not be `u32::MAX` (i.e. a ≠ 0).
#[inline]
pub fn re_char(dlog_a: u32, r: usize, order: usize) -> f64 {
    let phase = std::f64::consts::TAU * (r as f64) * (dlog_a as f64) / (order as f64);
    phase.cos()
}

/// Imaginary part of χ_r(a): sin(2π · r · dlog_a / order).
#[inline]
pub fn im_char(dlog_a: u32, r: usize, order: usize) -> f64 {
    let phase = std::f64::consts::TAU * (r as f64) * (dlog_a as f64) / (order as f64);
    phase.sin()
}

// ---------------------------------------------------------------------------
// Amplitude and correlation helpers
// ---------------------------------------------------------------------------

/// Complex amplitude |⟨u₁, χ_r(g(·))⟩| / (‖u₁‖ · √n_valid).
///
/// `dlogs_g[i]` = dlog_table[g(p_i)]; entries equal to `u32::MAX` are skipped.
///
/// This is the normalised Fourier coefficient of u₁ at frequency r in the
/// character basis.  For a perfect character eigenvector the amplitude = 1.
pub fn char_amplitude(u1: &[f64], dlogs_g: &[u32], r: usize, order: usize) -> f64 {
    debug_assert_eq!(u1.len(), dlogs_g.len());
    let mut re_sum = 0.0f64;
    let mut im_sum = 0.0f64;
    let mut u1_sq  = 0.0f64;
    let mut n_valid = 0usize;

    for (&u, &dlog) in u1.iter().zip(dlogs_g.iter()) {
        if dlog == u32::MAX {
            continue;
        }
        // ⟨u₁, χ̄_r⟩ uses conjugate: cos − i·sin
        re_sum += u * re_char(dlog, r, order);
        im_sum -= u * im_char(dlog, r, order); // minus for conjugate
        u1_sq  += u * u;
        n_valid += 1;
    }

    if n_valid == 0 || u1_sq < 1e-12 {
        return 0.0;
    }
    let amp_sq = re_sum * re_sum + im_sum * im_sum;
    amp_sq.sqrt() / (u1_sq.sqrt() * (n_valid as f64).sqrt())
}

/// Pearson correlation between two equal-length slices.
pub fn pearson_corr(x: &[f64], y: &[f64]) -> f64 {
    debug_assert_eq!(x.len(), y.len());
    let n = x.len();
    if n < 2 {
        return 0.0;
    }
    let n_f = n as f64;
    let mx = x.iter().sum::<f64>() / n_f;
    let my = y.iter().sum::<f64>() / n_f;
    let cov: f64 = x.iter().zip(y.iter()).map(|(a, b)| (a - mx) * (b - my)).sum();
    let sx: f64   = x.iter().map(|a| (a - mx).powi(2)).sum::<f64>().sqrt();
    let sy: f64   = y.iter().map(|b| (b - my).powi(2)).sum::<f64>().sqrt();
    if sx < 1e-12 || sy < 1e-12 {
        return 0.0;
    }
    cov / (sx * sy)
}

// ---------------------------------------------------------------------------
// Best-character search
// ---------------------------------------------------------------------------

/// Scan r = 0 … order/2 (inclusive) and return the r with maximum amplitude.
///
/// Returns `(r*, max_amplitude, top_chars)` where `top_chars` is a vec of
/// `(r, amplitude)` for the top-`top_n` characters by amplitude.
///
/// Using only r ≤ order/2 exploits the conjugate symmetry
/// |⟨u₁, χ_r⟩| = |⟨u₁, χ_{order−r}⟩| when u₁ is real.
pub fn best_character(
    u1: &[f64],
    dlogs_g: &[u32],
    order: usize,
    top_n: usize,
) -> (usize, f64, Vec<(usize, f64)>) {
    let half = order / 2;
    let mut best_r   = 0usize;
    let mut best_amp = 0.0f64;
    let mut all: Vec<(usize, f64)> = Vec::with_capacity(half + 1);

    for r in 0..=half {
        let amp = char_amplitude(u1, dlogs_g, r, order);
        all.push((r, amp));
        if amp > best_amp {
            best_amp = amp;
            best_r   = r;
        }
    }

    // Keep only top_n by amplitude
    all.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    all.truncate(top_n);

    (best_r, best_amp, all)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_primitive_root_small_primes() {
        // Known primitive roots:
        // 5  → 2 (order 4:  2^1=2, 2^2=4, 2^4≡1)
        // 7  → 3 (order 6)
        // 11 → 2 (order 10)
        // 691 → ? (we just verify it is a primitive root)
        for &ell in &[5u64, 7, 11, 13, 691, 131] {
            let g = primitive_root(ell);
            // Verify order is ell-1
            let order = ell - 1;
            let factors = distinct_prime_factors(order);
            for &p in &factors {
                assert_ne!(mod_pow(g, order / p, ell), 1,
                    "g={g} is not a primitive root of ell={ell}");
            }
            assert_eq!(mod_pow(g, order, ell), 1,
                "g^(ell-1) != 1 for g={g}, ell={ell}");
        }
    }

    #[test]
    fn test_dlog_table_roundtrip() {
        let ell = 7u64;
        let g   = primitive_root(ell);
        let tbl = build_dlog_table(ell, g);
        // g^tbl[a] ≡ a (mod ell) for all a in 1..ell
        for a in 1..ell as usize {
            let k = tbl[a];
            assert_ne!(k, u32::MAX);
            assert_eq!(mod_pow(g, k as u64, ell), a as u64,
                "dlog roundtrip failed at a={a}");
        }
    }

    #[test]
    fn test_trivial_character_is_one() {
        // χ_0(a) = cos(0) = 1 for all a
        for dlog in 0..10u32 {
            assert!((re_char(dlog, 0, 6) - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn test_char_amplitude_trivial_eigenvector() {
        // If u1 is a constant vector and all dlogs are 0 (g(p) = g^0 = 1 for all p),
        // then amplitude for r=0 should be 1 (trivial character).
        let n = 5usize;
        let u1: Vec<f64>      = vec![1.0 / (n as f64).sqrt(); n];
        let dlogs_g: Vec<u32> = vec![0; n]; // all dlog = 0 → g^0 = 1
        let order = 6usize;
        let amp0  = char_amplitude(&u1, &dlogs_g, 0, order);
        // Re(χ_0) = 1 everywhere, ⟨u1, χ̄_0⟩ = sum(u1) = sqrt(n)·(1/sqrt(n)) = 1
        // amplitude = 1 / (||u1|| · sqrt(n)) = 1/(1·sqrt(n)) * sqrt(n) = 1
        assert!((amp0 - 1.0).abs() < 1e-10, "amp0={amp0}");
    }

    #[test]
    fn test_pearson_corr_perfect() {
        let x: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
        let y: Vec<f64> = vec![2.0, 4.0, 6.0, 8.0]; // y = 2x
        let c = pearson_corr(&x, &y);
        assert!((c - 1.0).abs() < 1e-12, "c={c}");
    }

    #[test]
    fn test_pearson_corr_anticorrelated() {
        let x: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
        let y: Vec<f64> = vec![-1.0, -2.0, -3.0, -4.0];
        let c = pearson_corr(&x, &y);
        assert!((c + 1.0).abs() < 1e-12, "c={c}");
    }
}
