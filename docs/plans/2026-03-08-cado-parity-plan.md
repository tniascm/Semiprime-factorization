# CADO-NFS Parity Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Dominate CADO-NFS on c30 benchmark reproducibly (single-threaded CPU < 2000ms, multi-threaded wall < 1500ms).

**Architecture:** (1) Fix NFS parameters to match CADO's c30 params (log_i=9, decoupled sieve threshold). (2) Implement production-grade polynomial selection with non-monic leading coefficients, Murphy alpha scoring, linear rotation optimization, and Murphy E-value ranking. (3) Implement Block Wiedemann linear algebra with CSR sparse matrix and Berlekamp-Massey sequence solver.

**Tech Stack:** Rust, `rug` crate for multi-precision arithmetic (already a dependency). No new external dependencies.

**Reference:** CADO-NFS source at `/Users/andriipotapov/cado-nfs/`, parameter files at `/Users/andriipotapov/cado-nfs/parameters/factor/params.c*`.

---

## Phase 1: Parameter Alignment

### Task 1: Decouple sieve threshold from 2LP mfb bump

The 2LP merge bumps mfb from 18→36, which inflates the sieve threshold from 26→52, creating 230k false-positive survivors (vs ~30k needed). CADO uses mfb=18 for sieve threshold regardless of 2LP. Fix: store original mfb before 2LP bump and use it for sieve threshold.

**Files:**
- Modify: `rust/potapov-nfs/src/params.rs` (add `sieve_mfb0`, `sieve_mfb1` fields)
- Modify: `rust/potapov-nfs/src/pipeline.rs:526-545` (save original mfb before 2LP bump)
- Modify: `rust/potapov-nfs/src/sieve/mod.rs:326-327` (use sieve_mfb for threshold)

**Step 1: Write the failing test**

Add to `rust/potapov-nfs/src/params.rs` in the `#[cfg(test)] mod tests` block:

```rust
#[test]
fn test_sieve_mfb_defaults_to_mfb() {
    let p = NfsParams::c30();
    assert_eq!(p.sieve_mfb0, p.mfb0);
    assert_eq!(p.sieve_mfb1, p.mfb1);
}
```

**Step 2: Run test to verify it fails**

Run: `cd rust/potapov-nfs && cargo test test_sieve_mfb_defaults_to_mfb`
Expected: FAIL with "no field `sieve_mfb0`"

**Step 3: Add sieve_mfb fields to NfsParams**

In `rust/potapov-nfs/src/params.rs`, add two fields to the struct:

```rust
pub struct NfsParams {
    pub name: &'static str,
    pub degree: u32,
    pub lim0: u64,
    pub lim1: u64,
    pub lpb0: u32,
    pub lpb1: u32,
    pub mfb0: u32,
    pub mfb1: u32,
    pub sieve_mfb0: u32,
    pub sieve_mfb1: u32,
    pub log_i: u32,
    pub qmin: u64,
    pub qrange: u64,
    pub rels_wanted: u64,
}
```

Initialize `sieve_mfb0: 18, sieve_mfb1: 18` (same as mfb) in all presets (c30, c35, c40, c45). Fix all compilation errors from missing fields.

**Step 4: Save original mfb before 2LP bump**

In `rust/potapov-nfs/src/pipeline.rs`, BEFORE the 2LP bump block (line ~526), add:

```rust
params.sieve_mfb0 = params.mfb0;
params.sieve_mfb1 = params.mfb1;
```

This captures the pre-bump values. The 2LP bump then modifies `mfb0`/`mfb1` only, leaving `sieve_mfb0`/`sieve_mfb1` at the original values.

**Step 5: Use sieve_mfb for threshold computation**

In `rust/potapov-nfs/src/sieve/mod.rs`, change the threshold lines from:

```rust
let rat_bound = ((params.mfb0 as f64) * scale).min(255.0) as u8;
let alg_bound = ((params.mfb1 as f64) * scale).min(255.0) as u8;
```

to:

```rust
let rat_bound = ((params.sieve_mfb0 as f64) * scale).min(255.0) as u8;
let alg_bound = ((params.sieve_mfb1 as f64) * scale).min(255.0) as u8;
```

**Step 6: Run full test suite**

Run: `cd rust/potapov-nfs && cargo test --lib`
Expected: All tests pass.

**Step 7: Commit**

```bash
git add rust/potapov-nfs/src/params.rs rust/potapov-nfs/src/pipeline.rs rust/potapov-nfs/src/sieve/mod.rs
git commit -m "sieve: decouple sieve threshold from 2LP mfb bump"
```

---

### Task 2: Update c30 parameters to match CADO

CADO c30 uses `I=9` (our `log_i=9`) giving 16x more sieve area per q. Also update c35 parameters to match CADO's `params.c35`.

**Files:**
- Modify: `rust/potapov-nfs/src/params.rs` (update presets)

**Step 1: Write the failing test**

```rust
#[test]
fn test_c30_matches_cado() {
    let p = NfsParams::c30();
    assert_eq!(p.log_i, 9, "log_i should match CADO I=9");
    assert_eq!(p.lim0, 30_000);
    assert_eq!(p.lpb0, 17);
    assert_eq!(p.mfb0, 18);
}
```

**Step 2: Run test to verify it fails**

Run: `cd rust/potapov-nfs && cargo test test_c30_matches_cado`
Expected: FAIL with "log_i should match CADO I=9"

**Step 3: Update c30 preset**

In `rust/potapov-nfs/src/params.rs`, change `c30()`:

```rust
pub fn c30() -> Self {
    Self {
        name: "c30",
        degree: 3,
        lim0: 30_000,
        lim1: 30_000,
        lpb0: 17,
        lpb1: 17,
        mfb0: 18,
        mfb1: 18,
        sieve_mfb0: 18,
        sieve_mfb1: 18,
        log_i: 9,
        qmin: 50_000,
        qrange: 1_000,
        rels_wanted: 30_000,
    }
}
```

Also update c35 to match CADO's `params.c35`:

```rust
pub fn c35() -> Self {
    Self {
        name: "c35",
        degree: 3,
        lim0: 40_000,
        lim1: 40_000,
        lpb0: 18,
        lpb1: 18,
        mfb0: 20,
        mfb1: 20,
        sieve_mfb0: 20,
        sieve_mfb1: 20,
        log_i: 9,
        qmin: 25_000,
        qrange: 5_000,
        rels_wanted: 35_000,
    }
}
```

Update existing test `test_c30_params` to match new `log_i=9` and `sieve_half_width() = 512`.

**Step 4: Run full test suite**

Run: `cd rust/potapov-nfs && cargo test --lib`
Expected: All tests pass.

**Step 5: Benchmark**

Run: `RAYON_NUM_THREADS=1 cargo run --release -- --factor 684217602914977371691118975023 --threads 1`

Expected: Significant reduction in special-q count (from 2704 to ~200-500) and total time reduction.

**Step 6: Commit**

```bash
git add rust/potapov-nfs/src/params.rs
git commit -m "params: align c30/c35 with CADO parameter files"
```

---

## Phase 2: Production Polynomial Selection

### Task 3: Murphy alpha computation

Implement the Murphy alpha function that measures a polynomial's "root property" — how often small primes divide polynomial values. This is the core quality metric for polynomial selection.

Formula: `alpha(f) = sum_{p prime <= B} (1 - q_p * p / (p+1)) * log(p) / (p-1)`
where `q_p` = number of roots of f(x) mod p (affine) + projective contribution.

Reference: `/Users/andriipotapov/cado-nfs/polyselect/polyselect_alpha.c` lines 210-264.

**Files:**
- Create: `rust/gnfs/src/polyselect/alpha.rs`
- Modify: `rust/gnfs/src/polyselect.rs` (add `mod alpha; pub use alpha::murphy_alpha;`)

Note: `rust/gnfs/src/polyselect.rs` is currently a single file. Convert it to a module directory:
- Move `rust/gnfs/src/polyselect.rs` to `rust/gnfs/src/polyselect/mod.rs`
- Create `rust/gnfs/src/polyselect/alpha.rs`

**Step 1: Write the failing test**

Create `rust/gnfs/src/polyselect/alpha.rs` with test:

```rust
use rug::Integer;

/// Compute Murphy's alpha for polynomial f over primes up to `bound`.
///
/// Alpha measures the "root property": how often small primes divide f(x,y).
/// More negative alpha = better polynomial (more smooth values).
///
/// Formula: alpha(f) = sum_{p <= B} (1 - q_p * p/(p+1)) * log(p)/(p-1)
/// where q_p = number of roots of f mod p (affine + projective).
pub fn murphy_alpha(f_coeffs: &[i64], bound: u64) -> f64 {
    todo!()
}

/// Count the number of roots of f(x) mod p, including projective roots.
fn count_roots_mod_p(f_coeffs: &[i64], p: u64) -> u64 {
    todo!()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alpha_linear_polynomial() {
        // f(x) = x - 1 has exactly 1 root mod every prime (root = 1)
        // alpha should be negative (good root property)
        let alpha = murphy_alpha(&[-1, 1], 200);
        assert!(alpha < 0.0, "Linear poly should have negative alpha: {}", alpha);
    }

    #[test]
    fn test_alpha_irreducible_poly() {
        // f(x) = x^2 + 1 has 0 roots mod p when p = 3 mod 4, 2 roots when p = 1 mod 4
        // Overall alpha should be near 0 (balanced)
        let alpha = murphy_alpha(&[1, 0, 1], 200);
        assert!(alpha.abs() < 2.0, "x^2+1 alpha should be moderate: {}", alpha);
    }

    #[test]
    fn test_alpha_many_roots_is_better() {
        // f(x) = x*(x-1)*(x-2) = x^3 - 3x^2 + 2x has 3 roots mod most primes
        let alpha_good = murphy_alpha(&[0, 2, -3, 1], 200);
        // f(x) = x^3 + x + 1 has fewer roots mod most primes
        let alpha_bad = murphy_alpha(&[1, 1, 0, 1], 200);
        assert!(alpha_good < alpha_bad,
            "Poly with more roots should have more negative alpha: {} vs {}",
            alpha_good, alpha_bad);
    }

    #[test]
    fn test_count_roots_mod_p() {
        // x^2 - 1 = (x-1)(x+1) has 2 roots mod 5: x=1, x=4
        assert_eq!(count_roots_mod_p(&[-1, 0, 1], 5), 2);
        // x^2 + 1 has 0 roots mod 3 (QNR)
        assert_eq!(count_roots_mod_p(&[1, 0, 1], 3), 0);
        // x^2 + 1 has 2 roots mod 5: x=2, x=3
        assert_eq!(count_roots_mod_p(&[1, 0, 1], 5), 2);
        // x has 1 root mod any prime: x=0
        assert_eq!(count_roots_mod_p(&[0, 1], 7), 1);
    }

    #[test]
    fn test_projective_root() {
        // f(x) = 2*x^2 + x + 1: leading coeff 2, so projective root at p=2
        // Homogeneous: F(x,y) = 2x^2 + xy + y^2. At (1,0): F=2, divisible by 2.
        let roots = count_roots_mod_p(&[1, 1, 2], 2);
        // x=0: f(0)=1 mod 2 = 1 (not root). x=1: f(1)=4 mod 2 = 0 (root).
        // Plus projective: p | leading coeff. Total = 2.
        assert_eq!(roots, 2);
    }
}
```

**Step 2: Run test to verify it fails**

Run: `cd rust/gnfs && cargo test polyselect::alpha::tests`
Expected: FAIL with "not yet implemented"

**Step 3: Implement murphy_alpha and count_roots_mod_p**

```rust
use crate::arith::sieve_primes;

/// Compute Murphy's alpha for polynomial f over primes up to `bound`.
pub fn murphy_alpha(f_coeffs: &[i64], bound: u64) -> f64 {
    if f_coeffs.is_empty() {
        return 0.0;
    }
    let primes = sieve_primes(bound);
    let mut alpha = 0.0;
    for &p in &primes {
        let pf = p as f64;
        let q_p = count_roots_mod_p(f_coeffs, p) as f64;
        // CADO formula: (1 - q_p * p / (p+1)) * log(p) / (p-1)
        alpha += (1.0 - q_p * pf / (pf + 1.0)) * pf.ln() / (pf - 1.0);
    }
    alpha
}

/// Count roots of f(x) mod p (affine + projective).
fn count_roots_mod_p(f_coeffs: &[i64], p: u64) -> u64 {
    let p_i64 = p as i64;
    let mut count = 0u64;

    // Affine roots: evaluate f(x) mod p for x in 0..p
    for x in 0..p {
        let mut val = 0i64;
        let mut x_pow = 1i64;
        for &c in f_coeffs {
            val = (val + (c % p_i64) * x_pow) % p_i64;
            x_pow = (x_pow * (x as i64)) % p_i64;
        }
        if ((val % p_i64) + p_i64) % p_i64 == 0 {
            count += 1;
        }
    }

    // Projective root: check if p divides leading coefficient
    let d = f_coeffs.len() - 1;
    if d > 0 && f_coeffs[d] % p_i64 == 0 {
        count += 1;
    }

    count
}
```

Note: The affine root counting via exhaustive evaluation is O(p * deg) per prime. For `bound=2000`, the largest prime is ~2000, and we have ~300 primes. Total: ~300 * 2000 * 5 = 3M operations — fast enough (< 10ms). For production at c60+ with larger bounds, use Cantor-Zassenhaus factorization instead.

**Step 4: Run tests**

Run: `cd rust/gnfs && cargo test polyselect::alpha::tests`
Expected: All pass.

**Step 5: Wire into polyselect module**

Convert `rust/gnfs/src/polyselect.rs` to module directory if not already done. Add `mod alpha; pub use alpha::murphy_alpha;` to `mod.rs`.

**Step 6: Commit**

```bash
git add rust/gnfs/src/polyselect/
git commit -m "polyselect: add Murphy alpha computation for polynomial quality scoring"
```

---

### Task 4: Non-monic polynomial generation with ad sweep

Generate polynomials with arbitrary leading coefficient `ad` (not just 1). For degree d: `f(x) = ad*x^d + c_{d-1}*x^{d-1} + ... + c_0` where `ad*m^d + ... + c_0 = N`.

Reference: CADO `polyselect/polyselect.c` lines 215-270.

**Files:**
- Modify: `rust/gnfs/src/polyselect/mod.rs` (add `select_polynomial_with_ad`)

**Step 1: Write the failing test**

```rust
#[test]
fn test_select_polynomial_with_ad() {
    let n = Integer::from_str_radix("684217602914977371691118975023", 10).unwrap();
    // ad=1 should produce the same as the monic base-m
    let poly1 = select_polynomial_with_ad(&n, 3, 1);
    assert_eq!(poly1.degree, 3);
    // Verify f(m) = N: the polynomial evaluates correctly
    let m: Integer = poly1.m_str.parse().unwrap();
    let coeffs: Vec<Integer> = poly1.f_coeffs_str.iter()
        .map(|s| s.parse::<Integer>().unwrap()).collect();
    let mut val = Integer::from(0);
    let mut m_pow = Integer::from(1);
    for c in &coeffs {
        val += c * &m_pow;
        m_pow *= &m;
    }
    assert_eq!(val, n, "f(m) must equal N");
}

#[test]
fn test_nonmonic_polynomial_ad_60() {
    let n = Integer::from_str_radix("684217602914977371691118975023", 10).unwrap();
    let poly = select_polynomial_with_ad(&n, 3, 60);
    assert_eq!(poly.degree, 3);
    // Leading coefficient should be 60
    let lead: Integer = poly.f_coeffs_str.last().unwrap().parse().unwrap();
    assert_eq!(lead, Integer::from(60));
    // Verify f(m) = N
    let m: Integer = poly.m_str.parse().unwrap();
    let coeffs: Vec<Integer> = poly.f_coeffs_str.iter()
        .map(|s| s.parse::<Integer>().unwrap()).collect();
    let mut val = Integer::from(0);
    let mut m_pow = Integer::from(1);
    for c in &coeffs {
        val += c * &m_pow;
        m_pow *= &m;
    }
    assert_eq!(val, n, "f(m) must equal N for ad=60");
}
```

**Step 2: Run test to verify it fails**

Run: `cd rust/gnfs && cargo test test_nonmonic_polynomial`
Expected: FAIL with "cannot find function `select_polynomial_with_ad`"

**Step 3: Implement select_polynomial_with_ad**

```rust
/// Generate a polynomial pair with given leading coefficient `ad`.
///
/// Computes m = floor((N/ad)^(1/d)), then constructs f(x) = ad*x^d + ... + c_0
/// via base-m expansion such that f(m) = N. Returns None if ad doesn't yield
/// a valid polynomial (e.g., m too small).
pub fn select_polynomial_with_ad(n: &Integer, degree: u32, ad: u64) -> Option<PolynomialPair> {
    let d = degree as usize;
    let ad_int = Integer::from(ad);

    // m = floor((N / ad)^(1/d))
    let n_over_ad = Integer::from(n / &ad_int);
    let m = nth_root(&n_over_ad, degree);

    if m < 2 {
        return None;
    }

    // Express N in base m with leading coefficient ad:
    // N = ad*m^d + c_{d-1}*m^{d-1} + ... + c_0
    // Compute: remainder = N - ad*m^d, then extract coefficients via base-m expansion
    let m_pow_d = Integer::from(m.clone().pow(degree));
    let mut remaining = Integer::from(n - &ad_int * &m_pow_d);

    if remaining < 0 {
        // ad*m^d > N, try m-1
        let m = Integer::from(&m - 1);
        let m_pow_d = Integer::from(m.clone().pow(degree));
        remaining = Integer::from(n - &ad_int * &m_pow_d);
        if remaining < 0 {
            return None;
        }
        return build_poly_from_remainder(n, &m, &ad_int, d, &remaining);
    }

    build_poly_from_remainder(n, &m, &ad_int, d, &remaining)
}

fn build_poly_from_remainder(
    n: &Integer,
    m: &Integer,
    ad: &Integer,
    d: usize,
    remainder: &Integer,
) -> Option<PolynomialPair> {
    let mut coeffs = Vec::with_capacity(d + 1);
    let mut rem = remainder.clone();

    if *m <= 1 {
        return None;
    }

    // Extract d coefficients c_0 through c_{d-1} via base-m expansion
    for _ in 0..d {
        let (quot, r) = rem.div_rem_euc(m.clone());
        coeffs.push(r);
        rem = quot;
    }

    // rem should be 0 if ad*m^d was computed correctly
    // If not, adjust: the last coefficient absorbs any remainder
    if rem != 0 {
        // This means our expansion doesn't fit cleanly. Skip this ad.
        return None;
    }

    coeffs.push(ad.clone());

    // Verify: f(m) = N
    debug_assert!({
        let mut val = Integer::from(0);
        let mut m_pow = Integer::from(1);
        for c in &coeffs {
            val += c * &m_pow;
            m_pow *= m;
        }
        val == *n
    });

    let neg_m = Integer::from(-m);
    Some(PolynomialPair::new(&coeffs, &neg_m, &Integer::from(1), m, n))
}
```

**Step 4: Run tests**

Run: `cd rust/gnfs && cargo test test_nonmonic_polynomial`
Expected: All pass.

**Step 5: Commit**

```bash
git add rust/gnfs/src/polyselect/
git commit -m "polyselect: add non-monic polynomial generation with ad sweep"
```

---

### Task 5: Linear rotation optimization

Apply rotations `f'(x) = f(x) + (u*x + v) * g(x)` to improve root properties while preserving f(m) = N (since g(m) = 0).

Reference: `/Users/andriipotapov/cado-nfs/polyselect/ropt_arith.c` lines 157-166.

**Files:**
- Create: `rust/gnfs/src/polyselect/rotation.rs`

**Step 1: Write the failing test**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use rug::Integer;

    #[test]
    fn test_rotation_preserves_identity() {
        // f(x) = x^3 + 3x^2 + 5x + 7, g(x) = x - 10, m = 10
        // f(m) = 1000 + 300 + 50 + 7 = 1357
        // After rotation f' = f + v*g, f'(m) = f(m) + v*g(m) = f(m) + 0 = f(m)
        let f = vec![7i64, 5, 3, 1];
        let g = vec![-10i64, 1];
        let f_rot = apply_rotation(&f, &g, 0, 5); // f + 5*g
        // f + 5*g = (7 + 5*(-10)), (5 + 5*1), 3, 1 = [-43, 10, 3, 1]
        assert_eq!(f_rot, vec![-43, 10, 3, 1]);
    }

    #[test]
    fn test_rotation_with_ux() {
        // f + (3x + 0)*g where g = [g0, g1]
        // = f + 3*x*g = f[0], f[1]+3*g[0], f[2]+3*g[1], f[3]
        let f = vec![7i64, 5, 3, 1];
        let g = vec![-10i64, 1];
        let f_rot = apply_rotation(&f, &g, 3, 0); // f + 3*x*g
        assert_eq!(f_rot, vec![7, 5 + 3*(-10), 3 + 3*1, 1]);
    }

    #[test]
    fn test_optimize_rotation_improves_alpha() {
        let f = vec![7i64, 5, 3, 1];
        let g = vec![-10i64, 1];
        let alpha_before = crate::polyselect::alpha::murphy_alpha(&f, 200);
        let (best_f, best_u, best_v) = optimize_rotation(&f, &g, 50, 200);
        let alpha_after = crate::polyselect::alpha::murphy_alpha(&best_f, 200);
        // Optimized rotation should have equal or better alpha
        assert!(alpha_after <= alpha_before + 0.1,
            "Rotation should not worsen alpha: {} -> {}", alpha_before, alpha_after);
    }
}
```

**Step 2: Run test to verify it fails**

Run: `cd rust/gnfs && cargo test polyselect::rotation::tests`
Expected: FAIL

**Step 3: Implement rotation and optimization**

```rust
use crate::polyselect::alpha::murphy_alpha;

/// Apply rotation: f'(x) = f(x) + (u*x + v) * g(x)
///
/// This preserves f(m) = N since g(m) = 0.
/// g is assumed to be linear: g(x) = g[0] + g[1]*x.
pub fn apply_rotation(f: &[i64], g: &[i64], u: i64, v: i64) -> Vec<i64> {
    assert!(g.len() == 2, "g must be linear for rotation");
    let mut result = f.to_vec();

    // Add v*g: result[i] += v * g[i]
    if v != 0 {
        for (i, &gi) in g.iter().enumerate() {
            if i < result.len() {
                result[i] += v * gi;
            }
        }
    }

    // Add u*x*g: result[i+1] += u * g[i]
    if u != 0 {
        // Ensure result has enough space (degree may increase by 1)
        while result.len() < g.len() + 1 {
            result.push(0);
        }
        for (i, &gi) in g.iter().enumerate() {
            result[i + 1] += u * gi;
        }
    }

    // Trim trailing zeros (but keep at least degree+1 coefficients)
    while result.len() > 1 && *result.last().unwrap() == 0 {
        result.pop();
    }

    result
}

/// Search for the best rotation (u, v) minimizing avg_log_norm + murphy_alpha.
///
/// Tries u in [-u_range, u_range] and v in [-v_range, v_range].
/// Returns (best_f, best_u, best_v).
pub fn optimize_rotation(
    f: &[i64],
    g: &[i64],
    search_range: i64,
    alpha_bound: u64,
) -> (Vec<i64>, i64, i64) {
    let mut best_f = f.to_vec();
    let mut best_score = murphy_alpha(f, alpha_bound);
    let mut best_u = 0i64;
    let mut best_v = 0i64;

    // For degree 3 polynomials on c30-c45, the rotation search space is small.
    // v affects c_0 and c_1; u affects c_1 and c_2.
    // Search v first (affects lower coefficients more), then u.
    let v_range = search_range;
    let u_range = search_range / 10; // u rotations affect higher coefficients

    for v in -v_range..=v_range {
        let f_v = apply_rotation(f, g, 0, v);
        let score_v = murphy_alpha(&f_v, alpha_bound);

        if score_v < best_score {
            best_score = score_v;
            best_f = f_v.clone();
            best_u = 0;
            best_v = v;
        }

        // For each v, try a few u values
        if u_range > 0 {
            for u in -u_range..=u_range {
                if u == 0 { continue; }
                let f_uv = apply_rotation(f, g, u, v);
                let score_uv = murphy_alpha(&f_uv, alpha_bound);
                if score_uv < best_score {
                    best_score = score_uv;
                    best_f = f_uv;
                    best_u = u;
                    best_v = v;
                }
            }
        }
    }

    (best_f, best_u, best_v)
}
```

**Step 4: Run tests**

Run: `cd rust/gnfs && cargo test polyselect::rotation::tests`
Expected: All pass.

**Step 5: Commit**

```bash
git add rust/gnfs/src/polyselect/rotation.rs rust/gnfs/src/polyselect/mod.rs
git commit -m "polyselect: add linear rotation optimization for root property improvement"
```

---

### Task 6: Dickman rho function

The Dickman rho function `rho(u)` gives the probability that a random integer near `x` is `x^{1/u}`-smooth. Required for Murphy E-value computation.

**Files:**
- Create: `rust/gnfs/src/polyselect/dickman.rs`

**Step 1: Write the failing test**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dickman_rho_trivial() {
        // rho(u) = 1 for u <= 1
        assert!((dickman_rho(0.5) - 1.0).abs() < 1e-10);
        assert!((dickman_rho(1.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_dickman_rho_known_values() {
        // rho(2) = 1 - ln(2) ≈ 0.30685
        assert!((dickman_rho(2.0) - 0.30685).abs() < 0.001);
        // rho(3) ≈ 0.04861
        assert!((dickman_rho(3.0) - 0.04861).abs() < 0.001);
        // rho(4) ≈ 0.00491
        assert!((dickman_rho(4.0) - 0.00491).abs() < 0.001);
        // rho(5) ≈ 0.000354
        assert!((dickman_rho(5.0) - 0.000354).abs() < 0.0001);
    }

    #[test]
    fn test_dickman_rho_monotone_decreasing() {
        for i in 1..20 {
            let u = i as f64;
            assert!(dickman_rho(u) >= dickman_rho(u + 0.5),
                "rho should be monotone decreasing at u={}", u);
        }
    }
}
```

**Step 2: Run test to verify it fails**

**Step 3: Implement Dickman rho via precomputed table + interpolation**

```rust
/// Compute Dickman's rho function via differential-delay equation.
///
/// For u <= 1: rho(u) = 1
/// For 1 < u <= 2: rho(u) = 1 - ln(u)
/// For u > 2: computed via numerical ODE integration with step size 1/N.
pub fn dickman_rho(u: f64) -> f64 {
    if u <= 1.0 {
        return 1.0;
    }
    if u <= 2.0 {
        return 1.0 - u.ln();
    }

    // Numerical integration of rho'(t) = -rho(t-1)/t
    // Use Euler method with step size h = 1/1000
    let h = 0.001;
    let n_steps = ((u - 1.0) / h).ceil() as usize;

    // Precompute rho on [1, u] with step h
    let mut rho_table: Vec<f64> = Vec::with_capacity(n_steps + 1);

    // Fill [1, 2] analytically
    let n_init = (1.0 / h).ceil() as usize;
    for i in 0..=n_init.min(n_steps) {
        let t = 1.0 + i as f64 * h;
        rho_table.push(if t <= 2.0 { 1.0 - t.ln() } else { 0.0 });
    }

    // Fill (2, u] via Euler integration: rho'(t) = -rho(t-1)/t
    for i in (n_init + 1)..=n_steps {
        let t = 1.0 + i as f64 * h;
        // rho(t-1) is at index i - n_init (offset by 1/h steps)
        let idx_prev = if i >= n_init { i - n_init } else { 0 };
        let rho_prev = if idx_prev < rho_table.len() {
            rho_table[idx_prev]
        } else {
            0.0
        };
        let rho_cur = rho_table.last().copied().unwrap_or(0.0);
        let derivative = -rho_prev / t;
        rho_table.push((rho_cur + derivative * h).max(0.0));
    }

    rho_table.last().copied().unwrap_or(0.0)
}
```

**Step 4: Run tests and verify accuracy**

Run: `cd rust/gnfs && cargo test polyselect::dickman::tests`

**Step 5: Commit**

```bash
git add rust/gnfs/src/polyselect/dickman.rs rust/gnfs/src/polyselect/mod.rs
git commit -m "polyselect: add Dickman rho function for smoothness probability"
```

---

### Task 7: Murphy E-value scoring and combined polynomial search

Integrate Murphy alpha, Dickman rho, and non-monic generation into a full polynomial search pipeline matching CADO's approach.

Murphy E = integral over angles of `rho(log|f(x,y)| / log(Bf)) * rho(log|g(x,y)| / log(Bg))`.

Reference: `/Users/andriipotapov/cado-nfs/polyselect/murphyE.cpp` lines 55-87.

**Files:**
- Create: `rust/gnfs/src/polyselect/murphy_e.rs`
- Modify: `rust/gnfs/src/polyselect/mod.rs` (add `select_best_polynomial` function)

**Step 1: Write the failing test**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_murphy_e_positive() {
        let f = vec![7i64, 5, 3, 1];
        let g = vec![-10i64, 1];
        let skewness = 1.0;
        let e = murphy_e(&f, &g, skewness, 1e5, 1e5, 200);
        assert!(e > 0.0, "Murphy E should be positive: {}", e);
    }

    #[test]
    fn test_murphy_e_better_poly_scores_higher() {
        // A polynomial with many roots (good alpha) should score higher
        let f_good = vec![0i64, 2, -3, 1]; // x(x-1)(x-2): 3 roots mod most primes
        let f_bad = vec![1i64, 1, 0, 1];   // x^3 + x + 1: fewer roots
        let g = vec![-10i64, 1];
        let e_good = murphy_e(&f_good, &g, 1.0, 1e5, 1e5, 200);
        let e_bad = murphy_e(&f_bad, &g, 1.0, 1e5, 1e5, 200);
        assert!(e_good > e_bad,
            "Poly with better roots should have higher E: {} vs {}", e_good, e_bad);
    }
}
```

**Step 2: Run test to verify it fails**

**Step 3: Implement Murphy E-value**

```rust
use crate::polyselect::alpha::murphy_alpha;
use crate::polyselect::dickman::dickman_rho;
use std::f64::consts::PI;

/// Compute Murphy's E-value for a polynomial pair (f, g).
///
/// E = (1/K) * sum_{i=0}^{K-1} rho(log|f(x_i,y_i)|/log(Bf)) * rho(log|g(x_i,y_i)|/log(Bg))
///
/// where (x_i, y_i) are sample points on an ellipse scaled by skewness.
pub fn murphy_e(
    f_coeffs: &[i64],
    g_coeffs: &[i64],
    skewness: f64,
    bf: f64,
    bg: f64,
    alpha_bound: u64,
) -> f64 {
    let k = 1000; // Number of integration points (matches CADO default)
    let area = 1e16_f64; // Sieving area scaling

    let x_scale = (area * skewness).sqrt();
    let y_scale = (area / skewness).sqrt();

    let alpha_f = murphy_alpha(f_coeffs, alpha_bound);
    let alpha_g = murphy_alpha(g_coeffs, alpha_bound);

    let log_bf = bf.ln();
    let log_bg = bg.ln();

    let mut e_sum = 0.0;

    for i in 0..k {
        let theta = PI / (k as f64) * (i as f64 + 0.5);
        let xi = x_scale * theta.cos();
        let yi = y_scale * theta.sin();

        let vf = eval_homogeneous_f64(f_coeffs, xi, yi).abs().ln() + alpha_f;
        let vg = eval_homogeneous_f64(g_coeffs, xi, yi).abs().ln() + alpha_g;

        let uf = vf / log_bf;
        let ug = vg / log_bg;

        if uf > 0.0 && ug > 0.0 {
            e_sum += dickman_rho(uf) * dickman_rho(ug);
        }
    }

    e_sum / k as f64
}

/// Evaluate homogeneous polynomial F(x, y) = c_0*y^d + c_1*x*y^{d-1} + ... + c_d*x^d
fn eval_homogeneous_f64(coeffs: &[i64], x: f64, y: f64) -> f64 {
    let d = coeffs.len() - 1;
    let mut result = 0.0;
    let mut x_pow = 1.0;
    let mut y_pow = y.powi(d as i32);
    let y_inv = if y.abs() > 1e-300 { x / y } else { 0.0 };

    for (i, &c) in coeffs.iter().enumerate() {
        result += c as f64 * x_pow * y_pow;
        x_pow *= x;
        if i < d {
            y_pow /= y;
        }
    }
    result
}

/// Compute optimal skewness for a polynomial (minimizes L2 lognorm).
pub fn optimal_skewness(f_coeffs: &[i64]) -> f64 {
    // For degree d: skewness ≈ (|c_0| / |c_d|)^(1/d)
    let d = f_coeffs.len() - 1;
    if d == 0 { return 1.0; }
    let c0 = (f_coeffs[0] as f64).abs().max(1.0);
    let cd = (f_coeffs[d] as f64).abs().max(1.0);
    (c0 / cd).powf(1.0 / d as f64)
}
```

**Step 4: Implement combined search in polyselect/mod.rs**

Add `select_best_polynomial` that sweeps over ad values, applies rotation, and ranks by Murphy E:

```rust
/// Select the best polynomial for N by searching over leading coefficients
/// and applying root optimization.
///
/// Parameters match CADO's polyselect:
/// - admax: maximum leading coefficient to try
/// - incr: step size for ad sweep
/// - ropteffort: rotation search effort (1.0 = full)
/// - nrkeep: number of top polynomials to retain
pub fn select_best_polynomial(
    n: &Integer,
    degree: u32,
    admax: u64,
    incr: u64,
    ropteffort: f64,
    nrkeep: usize,
) -> Vec<PolynomialPair> {
    let alpha_bound = 2000u64;
    let bf = 1e7;
    let bg = 5e6;
    let rotation_range = (50.0 * ropteffort) as i64;

    let mut candidates: Vec<(f64, PolynomialPair)> = Vec::new();

    // Also try the monic polynomial (ad=1) for comparison
    let monic = select_base_m_variant(n, degree, 0);
    let monic_coeffs: Vec<i64> = monic.f_coeffs_str.iter()
        .map(|s| s.parse::<i64>().unwrap_or(0)).collect();
    let monic_g: Vec<i64> = monic.g_coeffs_str.iter()
        .map(|s| s.parse::<i64>().unwrap_or(0)).collect();
    let skew = murphy_e::optimal_skewness(&monic_coeffs);
    let e = murphy_e::murphy_e(&monic_coeffs, &monic_g, skew, bf, bg, alpha_bound);
    candidates.push((e, monic));

    // Sweep over leading coefficients
    let mut ad = incr;
    while ad <= admax {
        if let Some(poly) = select_polynomial_with_ad(n, degree, ad) {
            let f_coeffs: Vec<i64> = poly.f_coeffs_str.iter()
                .map(|s| s.parse::<i64>().unwrap_or(0)).collect();
            let g_coeffs: Vec<i64> = poly.g_coeffs_str.iter()
                .map(|s| s.parse::<i64>().unwrap_or(0)).collect();

            // Apply rotation optimization
            let (rotated_f, _u, _v) = if rotation_range > 0 {
                rotation::optimize_rotation(&f_coeffs, &g_coeffs, rotation_range, alpha_bound)
            } else {
                (f_coeffs.clone(), 0, 0)
            };

            let skew = murphy_e::optimal_skewness(&rotated_f);
            let e = murphy_e::murphy_e(&rotated_f, &g_coeffs, skew, bf, bg, alpha_bound);

            // Build PolynomialPair with rotated coefficients
            let rotated_coeffs_int: Vec<Integer> = rotated_f.iter()
                .map(|&c| Integer::from(c)).collect();
            let g0: Integer = g_coeffs[0].into();
            let g1: Integer = g_coeffs[1].into();
            let m: Integer = poly.m_str.parse().unwrap();
            let rotated_poly = PolynomialPair::new(
                &rotated_coeffs_int, &g0, &g1, &m, n,
            );

            candidates.push((e, rotated_poly));
        }
        ad += incr;
    }

    // Sort by Murphy E (descending — higher is better)
    candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    candidates.truncate(nrkeep);

    candidates.into_iter().map(|(_, p)| p).collect()
}
```

**Step 5: Wire into pipeline**

In `rust/potapov-nfs/src/pipeline.rs`, replace the simple variant loop with `select_best_polynomial`. The pipeline currently tries 5 variants sequentially; replace with the new search that produces ranked polynomials.

**Step 6: Run full test suite + benchmark**

Run: `cd rust/potapov-nfs && cargo test --lib && cargo run --release -- --factor 684217602914977371691118975023 --threads 1`

**Step 7: Commit**

```bash
git add rust/gnfs/src/polyselect/ rust/potapov-nfs/src/pipeline.rs
git commit -m "polyselect: production polynomial search with Murphy E ranking and rotation"
```

---

## Phase 3: Block Wiedemann Linear Algebra

### Task 8: Sparse matrix representation (CSR format)

**Files:**
- Create: `rust/gnfs/src/sparse.rs`
- Modify: `rust/gnfs/src/lib.rs` (add `pub mod sparse;`)

**Step 1: Write tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::BitRow;

    #[test]
    fn test_csr_from_bitrows() {
        let mut r0 = BitRow::new(4);
        r0.set(0); r0.set(2);
        let mut r1 = BitRow::new(4);
        r1.set(1); r1.set(3);
        let csr = CsrMatrix::from_bitrows(&[r0, r1], 4);
        assert_eq!(csr.nrows, 2);
        assert_eq!(csr.ncols, 4);
        assert_eq!(csr.row_ptr, vec![0, 2, 4]);
        assert_eq!(csr.col_idx, vec![0, 2, 1, 3]);
    }

    #[test]
    fn test_spmv_gf2() {
        // Matrix: [[1,0,1,0],[0,1,0,1]]
        // x = [1,1,0,0] (as u64 bitmask = 0b11 = 3)
        // Ax = [1, 1] (bitmask = 0b11 = 3)
        let mut r0 = BitRow::new(4);
        r0.set(0); r0.set(2);
        let mut r1 = BitRow::new(4);
        r1.set(1); r1.set(3);
        let csr = CsrMatrix::from_bitrows(&[r0, r1], 4);
        let x = vec![3u64]; // bits: col0=1, col1=1
        let mut y = vec![0u64; 1]; // 2 rows fits in 1 word
        csr.spmv_block(&x, &mut y, 1);
        assert_eq!(y[0] & 0b11, 0b11); // both rows should be 1
    }
}
```

**Step 2: Implement CSR matrix**

```rust
/// Compressed Sparse Row matrix over GF(2).
pub struct CsrMatrix {
    pub nrows: usize,
    pub ncols: usize,
    /// row_ptr[i]..row_ptr[i+1] gives the range of column indices for row i.
    pub row_ptr: Vec<usize>,
    /// Column indices of nonzero entries.
    pub col_idx: Vec<u32>,
}

impl CsrMatrix {
    pub fn from_bitrows(rows: &[BitRow], ncols: usize) -> Self {
        let nrows = rows.len();
        let mut row_ptr = Vec::with_capacity(nrows + 1);
        let mut col_idx = Vec::new();
        row_ptr.push(0);
        for row in rows {
            for (wi, &word) in row.bits.iter().enumerate() {
                let mut w = word;
                while w != 0 {
                    let bit = w.trailing_zeros() as usize;
                    let c = wi * 64 + bit;
                    if c < ncols {
                        col_idx.push(c as u32);
                    }
                    w &= w - 1;
                }
            }
            row_ptr.push(col_idx.len());
        }
        Self { nrows, ncols, row_ptr, col_idx }
    }

    /// Sparse matrix-vector multiply over GF(2) with 64-wide blocking.
    ///
    /// x is a block of 64 column vectors packed as ncols words (bit i of x[j] = vector i, row j).
    /// y is the output: nrows words.
    /// y = A * x (over GF(2)).
    pub fn spmv_block(&self, x: &[u64], y: &mut [u64], _block_size: usize) {
        let nwords = (self.nrows + 63) / 64;
        assert!(y.len() >= nwords);

        for i in 0..self.nrows {
            let start = self.row_ptr[i];
            let end = self.row_ptr[i + 1];
            let mut val = 0u64;
            for &c in &self.col_idx[start..end] {
                let word_idx = c as usize / 64;
                let bit_idx = c as usize % 64;
                if word_idx < x.len() {
                    val ^= (x[word_idx] >> bit_idx) & 1;
                }
            }
            // Wait — this is wrong for block SpMV. Let me fix.
            // For block SpMV: x has block_cols words per column of the block.
            // Each word of x packs 64 RHS vectors.
            // We need: y[i] = XOR of x[col] for all col in row i's nonzeros.
            // This gives us 64 results simultaneously.
            let i_word = i / 64;
            let i_bit = i % 64;
            // Actually for proper block SpMV, x should be ncols u64 words,
            // each packing 64 independent vectors.
            // y[row] XOR= x[col] for each nonzero (row, col).
            // But y has nrows entries, not nrows/64.
            // Let me restructure.
        }
        // Restructured: x[j] packs 64 vectors at column j.
        // y[i] accumulates XOR of x[col] for cols in row i.
        y.iter_mut().for_each(|v| *v = 0);
        for i in 0..self.nrows {
            let start = self.row_ptr[i];
            let end = self.row_ptr[i + 1];
            let mut acc = 0u64;
            for &c in &self.col_idx[start..end] {
                acc ^= x[c as usize];
            }
            y[i] = acc;
        }
    }

    /// Transpose SpMV: y = A^T * x over GF(2) with 64-wide blocking.
    pub fn spmv_transpose_block(&self, x: &[u64], y: &mut [u64]) {
        y.iter_mut().for_each(|v| *v = 0);
        for i in 0..self.nrows {
            let start = self.row_ptr[i];
            let end = self.row_ptr[i + 1];
            for &c in &self.col_idx[start..end] {
                y[c as usize] ^= x[i];
            }
        }
    }
}
```

**Step 3: Run tests, commit**

---

### Task 9: Berlekamp-Massey over GF(2) matrices

Find the minimal polynomial of a sequence of 64x64 GF(2) matrices. This is the core algebraic step of Block Wiedemann.

**Files:**
- Create: `rust/gnfs/src/berlekamp_massey.rs`

This is a substantial algorithm. Implement the block version that operates on sequences of u64 words (each representing a 64-bit vector over GF(2)).

Reference: Thomé, "Subquadratic computation of vector generating polynomials and computation of the Block Wiedemann algorithm", Journal of Symbolic Computation, 2002.

**Implementation note:** For c30 matrices (~8K), the sequence length is ~260 terms (2*8466/64). The BM step is O(n^2) in sequence length with 64-wide word ops.

---

### Task 10: Block Wiedemann solver

Combine CSR SpMV + Berlekamp-Massey into a complete Block Wiedemann nullspace finder.

**Files:**
- Create: `rust/gnfs/src/block_wiedemann.rs`
- Modify: `rust/gnfs/src/linalg.rs` (add `find_dependencies_bw`)

**Algorithm:**
1. Convert BitRow matrix to CSR
2. Generate random 64-column matrix X (nrows × 1 u64 words)
3. Generate random 64-column matrix Y (nrows × 1 u64 words)
4. Compute sequence: `S_i = X^T * A^i * Y` for i = 0..2*ceil(nrows/64)+10
5. Run Berlekamp-Massey on sequence to find minimal polynomial
6. Evaluate polynomial at A to get nullspace candidates
7. Extract individual dependency vectors

**Step 1: Write integration test**

```rust
#[test]
fn test_block_wiedemann_finds_dependency() {
    // Create a small matrix with known dependency
    let ncols = 10;
    let mut rows = Vec::new();
    // Row 0: cols 0, 1
    // Row 1: cols 1, 2
    // Row 2: cols 0, 2 (= row0 XOR row1)
    for (bits_set) in &[vec![0,1], vec![1,2], vec![0,2], vec![3,4], vec![4,5]] {
        let mut r = BitRow::new(ncols);
        for &b in bits_set { r.set(b); }
        rows.push(r);
    }
    let deps = find_dependencies_bw(&rows, ncols);
    assert!(!deps.is_empty(), "Should find at least one dependency");
    // Verify: XOR of rows in any dependency should be zero
    for dep in &deps {
        let mut xor = BitRow::new(ncols);
        for &r in dep { xor.xor_with(&rows[r]); }
        assert!(xor.is_zero(), "Dependency should XOR to zero");
    }
}
```

**Step 2: Implement and wire into pipeline**

In `rust/potapov-nfs/src/pipeline.rs`, update the LA gate:

```rust
let mut ge_deps = if matrix.len() > 5_000 {
    gnfs::linalg::find_dependencies_bw(&matrix, ncols)
} else {
    gnfs::linalg::find_dependencies(&matrix, ncols)
};
```

**Step 3: Benchmark against GE**

For c30 (8K matrix): BW should be ~2x faster than GE.
For c45 (50K+ matrix): BW should be ~10x faster than GE.

**Step 4: Commit**

```bash
git add rust/gnfs/src/sparse.rs rust/gnfs/src/berlekamp_massey.rs rust/gnfs/src/block_wiedemann.rs rust/gnfs/src/linalg.rs rust/gnfs/src/lib.rs rust/potapov-nfs/src/pipeline.rs
git commit -m "linalg: add Block Wiedemann solver with CSR sparse matrix"
```

---

## Phase 4: Integration and Benchmark

### Task 11: Full benchmark and parameter validation

**Step 1: Run 10-trial single-threaded benchmark**

```bash
for i in $(seq 1 10); do
    RAYON_NUM_THREADS=1 cargo run --release -- --factor 684217602914977371691118975023 --threads 1 2>&1 | grep -E '"total_ms"|"sieve_ms"|"la_ms"'
done
```

Target: total_ms < 2000.

**Step 2: Run 10-trial multi-threaded benchmark**

```bash
for i in $(seq 1 10); do
    cargo run --release -- --factor 684217602914977371691118975023 2>&1 | grep -E '"total_ms"|"sieve_ms"'
done
```

Target: total_ms < 1500.

**Step 3: Validate on multiple c30 semiprimes**

Test on at least 5 different c30 semiprimes to ensure reproducibility.

**Step 4: Update progress.md with final numbers**

**Step 5: Commit**

```bash
git add rust/potapov-nfs/progress.md
git commit -m "docs: update benchmarks post CADO-parity optimization"
```
