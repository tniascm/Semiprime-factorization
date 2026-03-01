# Rust-NFS Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a production NFS sieve in Rust that beats CADO-NFS on 96-128 bit semiprimes.

**Architecture:** New standalone crate `rust-nfs` implementing CADO's bucket sieve + special-q lattice sieve + cofactorization chain, reusing `gnfs` for LA and square root. All sieve arithmetic uses u64 Montgomery form. Parallelism via rayon.

**Tech Stack:** Rust, rayon, rug (for LA/sqrt big integers), gnfs crate (LA, sqrt, polynomial selection)

**Design Doc:** `docs/plans/2026-03-02-rust-nfs-design.md`

---

## Task 1: Scaffold Crate + Types + Parameters

**Files:**
- Create: `rust/rust-nfs/Cargo.toml`
- Create: `rust/rust-nfs/src/lib.rs`
- Create: `rust/rust-nfs/src/main.rs`
- Create: `rust/rust-nfs/src/params.rs`
- Create: `rust/rust-nfs/src/relation.rs`

**Step 1: Create Cargo.toml**

```toml
[package]
name = "rust-nfs"
version = "0.1.0"
edition = "2021"
description = "Production NFS implementation — bucket sieve + special-q lattice sieve"

[dependencies]
gnfs = { path = "../gnfs" }
rug = { version = "1", features = ["integer", "rand"] }
rayon = "1.10"
rand = "0.8"
serde = { version = "1", features = ["derive"] }
serde_json = "1"
clap = { version = "4", features = ["derive"] }

[profile.release]
lto = "fat"
codegen-units = 1
opt-level = 3
```

**Step 2: Create params.rs with CADO-matched parameters**

```rust
//! CADO-NFS-matched parameter tables for 96-200 bit semiprimes.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NfsParams {
    pub name: &'static str,
    pub degree: u32,
    pub lim0: u64,       // rational factor base bound
    pub lim1: u64,       // algebraic factor base bound
    pub lpb0: u32,       // rational large prime bound (bits)
    pub lpb1: u32,       // algebraic large prime bound (bits)
    pub mfb0: u32,       // rational max factor bound (bits)
    pub mfb1: u32,       // algebraic max factor bound (bits)
    pub log_i: u32,      // log2 of sieve region half-width
    pub qmin: u64,       // special-q start
    pub qrange: u64,     // special-q range per batch
    pub rels_wanted: u64,// target relation count
}

impl NfsParams {
    /// ~100 bits (30 decimal digits). From CADO params.c30.
    pub fn c30() -> Self {
        Self {
            name: "c30",
            degree: 3,
            lim0: 30_000, lim1: 30_000,
            lpb0: 17, lpb1: 17,
            mfb0: 18, mfb1: 18,
            log_i: 9,
            qmin: 50_000, qrange: 1_000,
            rels_wanted: 30_000,
        }
    }

    /// ~116 bits (35 decimal digits). From CADO params.c35.
    pub fn c35() -> Self {
        Self {
            name: "c35",
            degree: 3,
            lim0: 40_000, lim1: 40_000,
            lpb0: 18, lpb1: 18,
            mfb0: 20, mfb1: 20,
            log_i: 9,
            qmin: 55_000, qrange: 1_500,
            rels_wanted: 35_000,
        }
    }

    /// ~133 bits (40 decimal digits). From CADO params.c40.
    pub fn c40() -> Self {
        Self {
            name: "c40",
            degree: 4,
            lim0: 50_000, lim1: 55_000,
            lpb0: 18, lpb1: 18,
            mfb0: 22, mfb1: 22,
            log_i: 9,
            qmin: 55_000, qrange: 1_500,
            rels_wanted: 40_000,
        }
    }

    /// ~150 bits (45 decimal digits). From CADO params.c45.
    pub fn c45() -> Self {
        Self {
            name: "c45",
            degree: 4,
            lim0: 55_000, lim1: 65_000,
            lpb0: 18, lpb1: 19,
            mfb0: 24, mfb1: 26,
            log_i: 10,
            qmin: 58_000, qrange: 2_000,
            rels_wanted: 50_000,
        }
    }

    pub fn for_bits(bits: u32) -> Self {
        match bits {
            0..=105 => Self::c30(),
            106..=120 => Self::c35(),
            121..=140 => Self::c40(),
            _ => Self::c45(),
        }
    }

    pub fn large_prime_bound_0(&self) -> u64 { 1u64 << self.lpb0 }
    pub fn large_prime_bound_1(&self) -> u64 { 1u64 << self.lpb1 }
    pub fn sieve_half_width(&self) -> u64 { 1u64 << self.log_i }
    pub fn sieve_width(&self) -> u64 { 1u64 << (self.log_i + 1) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_params_c30() {
        let p = NfsParams::c30();
        assert_eq!(p.degree, 3);
        assert_eq!(p.large_prime_bound_0(), 131072); // 2^17
        assert_eq!(p.sieve_half_width(), 512);       // 2^9
    }

    #[test]
    fn test_params_for_bits() {
        assert_eq!(NfsParams::for_bits(96).name, "c30");
        assert_eq!(NfsParams::for_bits(112).name, "c35");
        assert_eq!(NfsParams::for_bits(128).name, "c40");
    }
}
```

**Step 3: Create relation.rs**

```rust
//! Relation types for NFS sieve output.
//!
//! Compatible with gnfs::types::Relation for handoff to LA/sqrt phases.

use serde::{Deserialize, Serialize};

/// A relation found by the sieve: (a, b) with factored norms.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Relation {
    pub a: i64,
    pub b: u64,
    /// Rational side: (prime_index_in_fb, exponent)
    pub rational_factors: Vec<(u32, u8)>,
    /// Algebraic side: (flat_index, exponent)
    pub algebraic_factors: Vec<(u32, u8)>,
    pub rational_sign_negative: bool,
    pub algebraic_sign_negative: bool,
    /// Rational large prime (0 if fully smooth)
    pub rat_cofactor: u64,
    /// Algebraic large prime (0 if fully smooth)
    pub alg_cofactor: u64,
}

impl Relation {
    pub fn is_full(&self) -> bool {
        self.rat_cofactor <= 1 && self.alg_cofactor <= 1
    }

    pub fn is_partial(&self) -> bool {
        !self.is_full()
    }

    /// Convert to gnfs::types::Relation for LA/sqrt phases.
    pub fn to_gnfs_relation(&self) -> gnfs::types::Relation {
        gnfs::types::Relation {
            a: self.a,
            b: self.b,
            rational_factors: self.rational_factors.clone(),
            algebraic_factors: self.algebraic_factors.clone(),
            rational_sign_negative: self.rational_sign_negative,
            algebraic_sign_negative: self.algebraic_sign_negative,
        }
    }
}
```

**Step 4: Create lib.rs and main.rs stubs**

```rust
// lib.rs
pub mod params;
pub mod relation;

pub use params::NfsParams;
pub use relation::Relation;
```

```rust
// main.rs
use clap::Parser;

#[derive(Parser)]
#[command(name = "rust-nfs", about = "Production NFS factorization")]
struct Cli {
    /// Number to factor (decimal)
    #[arg(long)]
    factor: Option<String>,

    /// Benchmark mode: bit sizes to test
    #[arg(long, value_delimiter = ',')]
    bits: Option<Vec<u32>>,

    /// Number of semiprimes per bit size
    #[arg(long, default_value = "3")]
    semiprimes: usize,

    /// Number of threads
    #[arg(long)]
    threads: Option<usize>,
}

fn main() {
    let cli = Cli::parse();
    if let Some(threads) = cli.threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
            .unwrap();
    }
    eprintln!("rust-nfs: not yet implemented");
}
```

**Step 5: Verify it compiles**

Run: `cd /Users/andriipotapov/Semiprime/rust/rust-nfs && cargo check`
Expected: Compiles with no errors.

**Step 6: Run tests**

Run: `cargo test -p rust-nfs`
Expected: 2 tests pass (params tests).

**Step 7: Commit**

```bash
git add rust/rust-nfs/
git commit -m "feat(rust-nfs): scaffold crate with CADO-matched params and relation types"
```

---

## Task 2: Montgomery u64 Arithmetic

Foundation for all sieve arithmetic: trial division, P-1, P+1, ECM.

**Files:**
- Create: `rust/rust-nfs/src/arith.rs`
- Modify: `rust/rust-nfs/src/lib.rs` — add `pub mod arith;`

**Step 1: Write failing tests for Montgomery arithmetic**

```rust
//! u64 Montgomery modular arithmetic.
//!
//! For modulus n < 2^63: represents x as x*R mod n where R = 2^64.
//! Multiplication: monty_mul(aR, bR) = abR mod n (one mul + one reduction).
//! No actual division anywhere — all reductions use precomputed n_inv.

/// Precomputed Montgomery parameters for a given odd modulus.
#[derive(Debug, Clone, Copy)]
pub struct MontgomeryParams {
    pub n: u64,
    /// -n^(-1) mod 2^64
    pub n_inv: u64,
    /// R mod n = 2^64 mod n
    pub r_mod_n: u64,
    /// R^2 mod n
    pub r2_mod_n: u64,
}

impl MontgomeryParams {
    pub fn new(n: u64) -> Self { todo!() }

    /// Convert a < n to Montgomery form: a*R mod n
    pub fn to_mont(&self, a: u64) -> u64 { todo!() }

    /// Convert from Montgomery form back to normal: aR^(-1) mod n
    pub fn from_mont(&self, ar: u64) -> u64 { todo!() }

    /// Montgomery multiplication: (aR * bR) * R^(-1) mod n = abR mod n
    pub fn mul(&self, ar: u64, br: u64) -> u64 { todo!() }

    /// Montgomery squaring (same as mul but may optimize)
    pub fn sqr(&self, ar: u64) -> u64 { todo!() }

    /// Montgomery modular exponentiation: a^e mod n (a in normal form)
    pub fn powmod(&self, a: u64, e: u64) -> u64 { todo!() }
}

/// Montgomery reduction: given t < n*2^64, compute t*R^(-1) mod n.
#[inline(always)]
fn monty_reduce(t: u128, n: u64, n_inv: u64) -> u64 { todo!() }

/// Precomputed trial divisor for fast divisibility testing.
/// Uses the identity: p | n iff n * p_inv <= p_lim (wrapping u64 mul).
#[derive(Debug, Clone, Copy)]
pub struct TrialDivisor {
    pub p: u64,
    pub p_inv: u64,   // p^(-1) mod 2^64
    pub p_lim: u64,   // floor((2^64 - 1) / p)
    pub log_p: u8,    // floor(log2(p) * scale)
}

impl TrialDivisor {
    pub fn new(p: u64, scale: f64) -> Self { todo!() }

    #[inline(always)]
    pub fn divides(&self, n: u64) -> bool {
        n.wrapping_mul(self.p_inv) <= self.p_lim
    }
}

/// Miller-Rabin primality test (1 round, base 2). Sufficient for NFS
/// cofactorization where false negatives just mean a missed relation.
pub fn is_probable_prime(n: u64) -> bool { todo!() }

/// Sieve of Eratosthenes up to bound.
pub fn sieve_primes(bound: u64) -> Vec<u64> { todo!() }

/// Extended GCD: returns (g, x) where g = gcd(a, m), a*x ≡ g (mod m).
pub fn extended_gcd(a: u64, m: u64) -> (u64, i64) { todo!() }

/// Modular inverse: a^(-1) mod m, or None if gcd(a,m) > 1.
pub fn mod_inverse(a: u64, m: u64) -> Option<u64> { todo!() }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_montgomery_roundtrip() {
        let mp = MontgomeryParams::new(97);
        for a in 0..97u64 {
            let ar = mp.to_mont(a);
            assert_eq!(mp.from_mont(ar), a, "roundtrip failed for a={}", a);
        }
    }

    #[test]
    fn test_montgomery_mul() {
        let mp = MontgomeryParams::new(97);
        for a in 1..97u64 {
            for b in 1..97u64 {
                let ar = mp.to_mont(a);
                let br = mp.to_mont(b);
                let cr = mp.mul(ar, br);
                assert_eq!(mp.from_mont(cr), (a * b) % 97,
                    "mul failed for {}*{} mod 97", a, b);
            }
        }
    }

    #[test]
    fn test_montgomery_powmod() {
        let mp = MontgomeryParams::new(97);
        // Fermat: a^96 ≡ 1 (mod 97) for a != 0
        for a in 1..97u64 {
            assert_eq!(mp.powmod(a, 96), 1, "Fermat failed for a={}", a);
        }
        // Known: 3^5 = 243 = 2*97 + 49
        assert_eq!(mp.powmod(3, 5), 49);
    }

    #[test]
    fn test_trial_divisor() {
        let td = TrialDivisor::new(7, 1.0);
        assert!(td.divides(0));
        assert!(td.divides(7));
        assert!(td.divides(14));
        assert!(td.divides(49));
        assert!(!td.divides(8));
        assert!(!td.divides(15));
        // Large value
        assert!(td.divides(7 * 1_000_000_007));
        assert!(!td.divides(7 * 1_000_000_007 + 1));
    }

    #[test]
    fn test_is_probable_prime() {
        assert!(!is_probable_prime(0));
        assert!(!is_probable_prime(1));
        assert!(is_probable_prime(2));
        assert!(is_probable_prime(3));
        assert!(!is_probable_prime(4));
        assert!(is_probable_prime(131071));   // Mersenne prime 2^17-1
        assert!(!is_probable_prime(131072));
        assert!(is_probable_prime(1_000_000_007));
    }

    #[test]
    fn test_mod_inverse() {
        assert_eq!(mod_inverse(3, 97), Some(65)); // 3*65 = 195 = 2*97 + 1
        assert_eq!(mod_inverse(2, 4), None);       // gcd(2,4) = 2
    }

    #[test]
    fn test_sieve_primes() {
        let primes = sieve_primes(30);
        assert_eq!(primes, vec![2, 3, 5, 7, 11, 13, 17, 19, 23, 29]);
    }

    #[test]
    fn test_montgomery_large_modulus() {
        // Test with a modulus near u64 max / 2
        let n = (1u64 << 62) - 57; // large prime
        let mp = MontgomeryParams::new(n);
        let a = n - 1;
        let ar = mp.to_mont(a);
        assert_eq!(mp.from_mont(ar), a);
        // a^2 mod n
        let a2 = mp.from_mont(mp.sqr(ar));
        assert_eq!(a2, 1); // (n-1)^2 = n^2 - 2n + 1 ≡ 1 (mod n)
    }
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p rust-nfs arith -- --nocapture`
Expected: FAIL — all `todo!()` functions panic.

**Step 3: Implement Montgomery arithmetic**

Implement all functions. Key algorithms:
- `n_inv`: Newton's method for -n^(-1) mod 2^64 (5 iterations from n itself)
- `monty_reduce`: REDC algorithm: m = (t mod R) * n_inv mod R; return (t + m*n) / R
- `to_mont`: monty_reduce(a as u128 * r2_mod_n as u128)
- `from_mont`: monty_reduce(ar as u128)
- `powmod`: binary method with Montgomery mul
- `is_probable_prime`: decompose n-1 = d*2^r, compute 2^d mod n, check witnesses
- `p_inv` for TrialDivisor: Newton's method for p^(-1) mod 2^64

**Step 4: Run tests to verify they pass**

Run: `cargo test -p rust-nfs arith`
Expected: All 8 tests pass.

**Step 5: Commit**

```bash
git add rust/rust-nfs/src/arith.rs
git commit -m "feat(rust-nfs): Montgomery u64 arithmetic + trial divisor + Miller-Rabin"
```

---

## Task 3: Factor Base Construction

**Files:**
- Create: `rust/rust-nfs/src/factorbase.rs`
- Modify: `rust/rust-nfs/src/lib.rs` — add `pub mod factorbase;`

**Step 1: Write failing tests**

```rust
//! Factor base construction: primes + polynomial roots + log table.

use crate::arith::{sieve_primes, TrialDivisor};

/// Factor base for one side of NFS.
#[derive(Debug, Clone)]
pub struct FactorBase {
    /// Primes up to the factor base bound.
    pub primes: Vec<u64>,
    /// For each prime p, all roots r of f(x) ≡ 0 (mod p).
    pub roots: Vec<Vec<u64>>,
    /// Precomputed trial divisors (Montgomery form).
    pub trial_divisors: Vec<TrialDivisor>,
    /// Quantized log2(p) values for sieve accumulation.
    pub log_p: Vec<u8>,
    /// Scale factor: log_p[i] = floor(log2(primes[i]) * scale).
    pub scale: f64,
}

impl FactorBase {
    /// Build factor base from polynomial coefficients and bound.
    /// `f_coeffs` are the polynomial coefficients [c0, c1, ..., cd].
    pub fn new(f_coeffs: &[i64], bound: u64, scale: f64) -> Self { todo!() }

    /// Total number of (prime, root) pairs.
    pub fn pair_count(&self) -> usize {
        self.roots.iter().map(|r| r.len()).sum()
    }
}

/// Find all roots of f(x) ≡ 0 (mod p) for prime p.
/// Uses exhaustive search for p < 1000, Tonelli-Shanks-based for larger.
pub fn find_roots_mod_p(f_coeffs: &[i64], p: u64) -> Vec<u64> { todo!() }

/// Evaluate polynomial f at x modulo m.
pub fn eval_poly_mod(f_coeffs: &[i64], x: u64, m: u64) -> u64 { todo!() }

/// Tonelli-Shanks: find r such that r^2 ≡ n (mod p), or None.
pub fn tonelli_shanks(n: u64, p: u64) -> Option<u64> { todo!() }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eval_poly_mod() {
        // f(x) = x^2 + 1
        let f = vec![1, 0, 1i64];
        assert_eq!(eval_poly_mod(&f, 3, 7), 3); // 9+1=10, 10%7=3
        assert_eq!(eval_poly_mod(&f, 0, 7), 1);
    }

    #[test]
    fn test_find_roots_small() {
        // f(x) = x^2 - 1 = (x-1)(x+1)
        let f = vec![-1, 0, 1i64];
        let roots = find_roots_mod_p(&f, 7);
        assert_eq!(roots.len(), 2);
        assert!(roots.contains(&1));
        assert!(roots.contains(&6)); // -1 mod 7 = 6
    }

    #[test]
    fn test_find_roots_degree3() {
        // f(x) = x^3 + 2x + 1
        let f = vec![1, 2, 0, 1i64];
        let roots = find_roots_mod_p(&f, 5);
        // Check each root: f(r) ≡ 0 (mod 5)
        for &r in &roots {
            assert_eq!(eval_poly_mod(&f, r, 5), 0, "r={} not a root mod 5", r);
        }
    }

    #[test]
    fn test_tonelli_shanks() {
        // 4 is a QR mod 7: sqrt(4) = 2 or 5
        let r = tonelli_shanks(4, 7).unwrap();
        assert!(r == 2 || r == 5);
        // 3 is a QNR mod 7
        assert!(tonelli_shanks(3, 7).is_none());
    }

    #[test]
    fn test_factor_base_construction() {
        // f(x) = x^3 + 2x + 1, bound = 50
        let f = vec![1, 2, 0, 1i64];
        let fb = FactorBase::new(&f, 50, 1.0);
        assert!(fb.primes.len() > 0);
        assert_eq!(fb.primes[0], 2);
        assert_eq!(fb.primes.len(), fb.roots.len());
        assert_eq!(fb.primes.len(), fb.trial_divisors.len());
        // Verify all roots are valid
        for (i, p) in fb.primes.iter().enumerate() {
            for &r in &fb.roots[i] {
                assert_eq!(eval_poly_mod(&f, r, *p), 0,
                    "root {} invalid for prime {}", r, p);
            }
        }
    }
}
```

**Step 2: Run tests to verify they fail, implement, verify pass, commit.**

---

## Task 4: Q-Lattice Reduction

**Files:**
- Create: `rust/rust-nfs/src/sieve/mod.rs`
- Create: `rust/rust-nfs/src/sieve/lattice.rs`
- Modify: `rust/rust-nfs/src/lib.rs` — add `pub mod sieve;`

**Step 1: Write failing tests for skew Gaussian lattice reduction**

```rust
//! Q-lattice and P-lattice reduction for special-q sieve.

/// Reduced basis for the q-lattice.
#[derive(Debug, Clone, Copy)]
pub struct QLattice {
    pub a0: i64, pub b0: i64,
    pub a1: i64, pub b1: i64,
}

/// Reduce the q-lattice for special-q prime q with root r.
/// Input basis: v0 = (q, 0), v1 = (r, 1).
/// Uses skew Gaussian reduction with quadratic form Q(a,b) = a^2 + skew^2 * b^2.
pub fn reduce_qlattice(q: u64, r: u64, skewness: f64) -> QLattice { todo!() }

/// Reduced p-lattice for Franke-Kleinjung enumeration.
#[derive(Debug, Clone, Copy)]
pub struct PLattice {
    /// Starting position in the sieve region.
    pub start: u64,
    /// Step increment (j1 * width + i1 in x-encoding).
    pub inc_step: i64,
    /// Warp increment (j0 * width + i0 in x-encoding).
    pub inc_warp: i64,
    /// Bound for step test.
    pub bound_step: i64,
    /// Bound for warp test.
    pub bound_warp: i64,
    /// Whether this prime hits the sieve region at all.
    pub hits: bool,
}

/// Reduce p-lattice within the q-lattice.
/// For FB prime p with root R (of the algebraic polynomial), find the reduced
/// basis for enumerating p's hits in the sieve region [-I, I) x [0, J).
pub fn reduce_plattice(p: u64, r: u64, qlat: &QLattice, log_i: u32) -> PLattice { todo!() }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qlattice_basic() {
        // q=97, r=30 (arbitrary root), skewness=1.0
        let ql = reduce_qlattice(97, 30, 1.0);
        // Verify: a0*b1 - a1*b0 = ±q (determinant of reduced basis = q)
        let det = (ql.a0 as i128 * ql.b1 as i128 - ql.a1 as i128 * ql.b0 as i128).abs();
        assert_eq!(det, 97, "determinant should be q=97");
        // Verify: the lattice contains (r, 1)
        // i.e., there exist integers u,v such that u*(a0,b0) + v*(a1,b1) = (r, 1)
        // Since det=q and v*b1 + u*b0 = 1, this system has a solution.
    }

    #[test]
    fn test_qlattice_short_vectors() {
        let ql = reduce_qlattice(65537, 12345, 1.0);
        let det = (ql.a0 as i128 * ql.b1 as i128 - ql.a1 as i128 * ql.b0 as i128).abs();
        assert_eq!(det, 65537);
        // Both vectors should be short (< sqrt(q) ≈ 256)
        let len0 = ((ql.a0 as f64).powi(2) + (ql.b0 as f64).powi(2)).sqrt();
        let len1 = ((ql.a1 as f64).powi(2) + (ql.b1 as f64).powi(2)).sqrt();
        assert!(len0 < 512.0, "v0 too long: {}", len0);
        assert!(len1 < 512.0, "v1 too long: {}", len1);
    }

    #[test]
    fn test_plattice_small_prime() {
        let ql = reduce_qlattice(97, 30, 1.0);
        let pl = reduce_plattice(7, 3, &ql, 9); // I=512
        // Prime 7 should have hits in the sieve region
        assert!(pl.hits, "prime 7 should hit sieve region");
    }
}
```

**Step 2: Implement skew Gaussian q-lattice reduction.**

Algorithm (from CADO `las-qlattice.cpp`):
```
v0 = (q, 0), v1 = (r, 1)
Q(a,b) = a^2 + S^2*b^2
loop:
  if Q(v0) > Q(v1): swap(v0, v1)
  mu = round((v0.a*v1.a + S^2*v0.b*v1.b) / Q(v0))
  v1 = v1 - mu*v0
  if v1 unchanged: break
```

**Step 3: Implement Franke-Kleinjung p-lattice reduction.**

Algorithm (from CADO `las-reduce-plattice-simplistic.hpp`):
```
Transform root R through q-lattice: R' = (R*a1 + b1) mod p (or similar)
Initial basis: (p, 0), (R', 1)
Partial GCD reduction until -I < i0 <= 0 <= i1 < I, i1 + |i0| >= I
Compute walk parameters: inc_step, inc_warp, bounds
```

**Step 4: Run tests, verify pass, commit.**

---

## Task 5: Bucket Sieve Data Structure

**Files:**
- Create: `rust/rust-nfs/src/sieve/bucket.rs`

**Step 1: Write failing tests**

```rust
//! Bucket sieve: cache-friendly sieve update accumulation.
//!
//! The sieve region is split into "bucket regions" of size 2^LOG_BUCKET_REGION.
//! For each large FB prime, instead of striding through the sieve array (cache-hostile),
//! we push compact 2-byte updates into the target bucket. Then for each bucket region
//! (which fits in L1 cache), we gather all updates and apply them locally.

/// Log2 of bucket region size. 16 = 64KB for x86, 17 = 128KB for Apple Silicon.
pub const LOG_BUCKET_REGION: u32 = 16;
pub const BUCKET_REGION: usize = 1 << LOG_BUCKET_REGION;

/// A compact sieve update: position within bucket region + log contribution.
#[derive(Debug, Clone, Copy)]
#[repr(C, packed)]
pub struct BucketUpdate {
    /// Position within the bucket region (0..BUCKET_REGION-1).
    pub pos: u16,
    /// Quantized log2(p) to subtract from sieve array.
    pub logp: u8,
}

/// Bucket array: pre-allocated storage for all bucket updates.
pub struct BucketArray {
    /// Flat storage for all updates across all buckets.
    data: Vec<BucketUpdate>,
    /// Per-bucket: current write position (index into data).
    write_pos: Vec<usize>,
    /// Per-bucket: start position in data.
    starts: Vec<usize>,
    /// Number of buckets.
    n_buckets: usize,
}

impl BucketArray {
    /// Allocate bucket array for `n_buckets` buckets with estimated `updates_per_bucket`.
    pub fn new(n_buckets: usize, updates_per_bucket: usize) -> Self { todo!() }

    /// Push an update into the specified bucket.
    #[inline(always)]
    pub fn push(&mut self, bucket: usize, update: BucketUpdate) { todo!() }

    /// Get all updates for a bucket region.
    pub fn updates_for_bucket(&self, bucket: usize) -> &[BucketUpdate] { todo!() }

    /// Clear all buckets for reuse with next special-q.
    pub fn clear(&mut self) { todo!() }

    /// Total number of updates across all buckets.
    pub fn total_updates(&self) -> usize { todo!() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bucket_push_and_read() {
        let mut ba = BucketArray::new(4, 100);
        ba.push(0, BucketUpdate { pos: 10, logp: 5 });
        ba.push(0, BucketUpdate { pos: 20, logp: 3 });
        ba.push(2, BucketUpdate { pos: 15, logp: 7 });

        let b0 = ba.updates_for_bucket(0);
        assert_eq!(b0.len(), 2);
        assert_eq!(b0[0].pos, 10);
        assert_eq!(b0[1].pos, 20);

        let b1 = ba.updates_for_bucket(1);
        assert_eq!(b1.len(), 0);

        let b2 = ba.updates_for_bucket(2);
        assert_eq!(b2.len(), 1);
        assert_eq!(b2[0].pos, 15);

        assert_eq!(ba.total_updates(), 3);
    }

    #[test]
    fn test_bucket_clear() {
        let mut ba = BucketArray::new(4, 100);
        ba.push(0, BucketUpdate { pos: 10, logp: 5 });
        ba.clear();
        assert_eq!(ba.updates_for_bucket(0).len(), 0);
        assert_eq!(ba.total_updates(), 0);
    }
}
```

**Step 2: Implement, test, commit.**

---

## Task 6: Norm Initialization + Small Sieve

**Files:**
- Create: `rust/rust-nfs/src/sieve/norm.rs`
- Create: `rust/rust-nfs/src/sieve/small.rs`

**Step 1: Implement u8 log-norm initialization**

For degree-1 (rational side): piecewise monotonic fill using transition points.
For degree-d (algebraic side): piecewise-linear approximation of log|F(i,j)|.

```rust
//! Norm initialization: fill sieve arrays with u8 log-norm approximations.

/// Initialize rational sieve array for row j.
/// Rational norm: |g1*a + g0*b| where a = a0*i + a1*j, b = b0*i + b1*j
/// (q-lattice transformed coordinates).
pub fn init_norm_rat(
    sieve: &mut [u8],
    g0: f64, g1: f64,   // rational poly coefficients
    a0: f64, b0: f64,   // q-lattice column 0
    a1: f64, b1: f64,   // q-lattice column 1
    j: u32,
    half_i: u32,         // I/2
    scale: f64,
    guard: u8,
) { todo!() }

/// Initialize algebraic sieve array for row j.
/// Uses Horner evaluation of log|F(a0*i+a1*j, b0*i+b1*j)| with f64 arithmetic.
pub fn init_norm_alg(
    sieve: &mut [u8],
    f_coeffs: &[f64],
    a0: f64, b0: f64,
    a1: f64, b1: f64,
    j: u32,
    half_i: u32,
    scale: f64,
    guard: u8,
) { todo!() }
```

**Step 2: Implement small sieve (line sieve for small primes)**

```rust
//! Small sieve: line sieve for primes smaller than the bucket threshold.

/// Sieve a row of the rational side for small primes.
/// For each prime p with known starting position, stride through the sieve
/// array subtracting log(p) at each hit.
pub fn small_sieve_rat(
    sieve: &mut [u8],
    primes: &[(u64, u8)],  // (prime, log_p)
    m_mod_p: &[u64],       // precomputed: m mod p for each prime
    b: u64,                 // current b-value
    half_i: u32,
    offset: usize,          // start offset within the sieve row
    length: usize,          // length of this chunk
) { todo!() }

/// Sieve a row of the algebraic side for small primes.
/// For each prime p and each root r: stride at step p from starting position.
pub fn small_sieve_alg(
    sieve: &mut [u8],
    primes: &[u64],
    roots: &[Vec<u64>],
    log_p: &[u8],
    b: u64,
    half_i: u32,
    offset: usize,
    length: usize,
) { todo!() }
```

**Step 3: Write tests verifying correct sieve accumulation on small examples.**

**Step 4: Run tests, verify pass, commit.**

---

## Task 7: Per-Region Processing + Survivor Collection

**Files:**
- Create: `rust/rust-nfs/src/sieve/region.rs`

**Step 1: Implement per-bucket-region processing**

This is the hot inner loop: init norms, apply bucket updates, small sieve,
scan for survivors.

```rust
//! Per-bucket-region processing: the inner loop of the NFS sieve.

use crate::sieve::bucket::{BucketArray, BUCKET_REGION};

/// A sieve survivor: position that passed the threshold test.
#[derive(Debug, Clone, Copy)]
pub struct Survivor {
    pub i: i64,  // a-coordinate
    pub j: u64,  // b-coordinate
}

/// Process one bucket region: apply updates, scan for survivors.
///
/// 1. Initialize u8 norm arrays (rational + algebraic)
/// 2. Apply bucket updates (large primes): saturated subtraction
/// 3. Apply small sieve (small primes): line sieve within region
/// 4. Scan: positions where BOTH rat_sieve[pos] + alg_sieve[pos] <= threshold
pub fn process_bucket_region(
    rat_sieve: &mut [u8; BUCKET_REGION],
    alg_sieve: &mut [u8; BUCKET_REGION],
    rat_updates: &[super::bucket::BucketUpdate],
    alg_updates: &[super::bucket::BucketUpdate],
    rat_bound: u8,
    alg_bound: u8,
    // ... small sieve parameters ...
) -> Vec<Survivor> { todo!() }

/// Apply bucket updates to sieve array using saturated subtraction.
/// This is critical for performance: the inner loop runs billions of times.
#[inline(always)]
pub fn apply_bucket_updates(sieve: &mut [u8], updates: &[super::bucket::BucketUpdate]) {
    for u in updates {
        let pos = u.pos as usize;
        // Saturated subtraction: never wraps below 0
        sieve[pos] = sieve[pos].saturating_sub(u.logp);
    }
}

/// Scan sieve array for survivors (positions where value <= bound).
/// Returns offsets within the bucket region.
pub fn scan_survivors(
    rat_sieve: &[u8],
    alg_sieve: &[u8],
    rat_bound: u8,
    alg_bound: u8,
) -> Vec<u16> {
    let mut survivors = Vec::new();
    for i in 0..rat_sieve.len().min(alg_sieve.len()) {
        if rat_sieve[i] <= rat_bound && alg_sieve[i] <= alg_bound {
            survivors.push(i as u16);
        }
    }
    survivors
}
```

**Step 2: Tests, implement, commit.**

---

## Task 8: Special-Q Sieve Loop

**Files:**
- Modify: `rust/rust-nfs/src/sieve/mod.rs` — main sieve orchestration

**Step 1: Implement the full special-q sieve loop**

```rust
//! Special-q lattice sieve: the main sieve entry point.

pub mod bucket;
pub mod lattice;
pub mod norm;
pub mod small;
pub mod region;

use crate::factorbase::FactorBase;
use crate::params::NfsParams;
use crate::relation::Relation;

/// Result of sieving: relations + timing breakdown.
#[derive(Debug, Clone)]
pub struct SieveResult {
    pub relations: Vec<Relation>,
    pub special_qs_processed: usize,
    pub survivors_found: usize,
    pub total_ms: f64,
    pub sieve_ms: f64,
    pub cofactor_ms: f64,
}

/// Run the special-q lattice sieve.
///
/// For each special-q prime q in [qmin, qmin+qrange):
///   1. Reduce q-lattice
///   2. For each FB prime: reduce p-lattice, scatter bucket updates
///   3. For each bucket region: apply updates, scan survivors
///   4. Cofactorize survivors
///
/// Parallelism: rayon parallel iteration over special-q primes.
pub fn sieve_specialq(
    poly: &gnfs::types::PolynomialPair,
    rat_fb: &FactorBase,
    alg_fb: &FactorBase,
    params: &NfsParams,
) -> SieveResult { todo!() }
```

**Step 2: Integration test — sieve a known small semiprime, verify relations are valid.**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sieve_small_semiprime() {
        // N = 8051 = 83 * 97
        let n = rug::Integer::from(8051u64);
        let poly = gnfs::polyselect::select_base_m(&n, 3);
        let f_i64 = gnfs::sieve::poly_coeffs_to_i64(&poly.f_coeffs()).unwrap();
        let params = NfsParams::c30(); // use c30 params
        let rat_fb = FactorBase::new(&[-1, 1], params.lim0, 1.442);
        let alg_fb = FactorBase::new(&f_i64, params.lim1, 1.442);

        let result = sieve_specialq(&poly, &rat_fb, &alg_fb, &params);
        // Should find at least SOME relations
        assert!(result.relations.len() > 0,
            "sieve should find relations for N=8051");
    }
}
```

**Step 3: Implement the full loop, integrating all previous modules.**

**Step 4: Run test, verify pass, commit.**

---

## Task 9: Cofactorization Pipeline (Trial Div + P-1 + P+1 + ECM)

**Files:**
- Create: `rust/rust-nfs/src/cofactor/mod.rs`
- Create: `rust/rust-nfs/src/cofactor/trialdiv.rs`
- Create: `rust/rust-nfs/src/cofactor/pm1.rs`
- Create: `rust/rust-nfs/src/cofactor/pp1.rs`
- Create: `rust/rust-nfs/src/cofactor/ecm.rs`
- Modify: `rust/rust-nfs/src/lib.rs` — add `pub mod cofactor;`

**Step 1: Implement Montgomery trial division**

```rust
//! Batch trial division using precomputed Montgomery inverses.

use crate::arith::TrialDivisor;

/// Trial divide n by all primes in the factor base.
/// Returns (factored_exponents, cofactor).
/// Uses Montgomery-form divisibility test: no actual division until a hit.
pub fn trial_divide(
    mut n: u64,
    divisors: &[TrialDivisor],
) -> (Vec<(u32, u8)>, u64) {
    let mut factors = Vec::new();
    for (i, td) in divisors.iter().enumerate() {
        if td.p * td.p > n { break; }
        if td.divides(n) {
            let mut exp = 0u8;
            while n % td.p == 0 {
                n /= td.p;
                exp += 1;
            }
            factors.push((i as u32, exp));
        }
    }
    (factors, n)
}
```

**Step 2: Implement P-1 method**

```rust
//! Pollard P-1 factoring method with u64 Montgomery arithmetic.

use crate::arith::MontgomeryParams;

/// P-1 Stage 1: compute 2^(lcm(1..B1)) mod n, check gcd.
/// Stage 2: check primes in (B1, B2].
pub fn pm1(n: u64, b1: u64, b2: u64) -> Option<u64> { todo!() }
```

**Step 3: Implement P+1 method**

```rust
//! Williams P+1 factoring method.

/// P+1 using Lucas sequence V_k(P, 1) mod n.
/// Starting value P = 2/7 (CADO default).
pub fn pp1(n: u64, b1: u64, b2: u64) -> Option<u64> { todo!() }
```

**Step 4: Implement ECM with u64 Montgomery arithmetic**

```rust
//! ECM with u64 Montgomery curve arithmetic.
//! For cofactors < 2^64, this is much faster than BigUint ECM.

/// Run one ECM curve with given B1, B2 bounds.
pub fn ecm_one_curve(n: u64, b1: u64, b2: u64, seed: u64) -> Option<u64> { todo!() }

/// ECM B1 sequence (CADO default): 105, 115, 126, 137, 149, ...
pub fn ecm_bounds(lpb: u32) -> Vec<(u64, u64)> {
    let ncurves = match lpb {
        0..=19 => 0,
        20..=22 => 1,
        23 => 2,
        24 => 4,
        25 => 5,
        26 => 6,
        27 => 8,
        28 => 11,
        _ => 16,
    };
    let mut bounds = Vec::new();
    let mut b1 = 105.0f64;
    for _ in 0..ncurves {
        let b2 = ((2.0 * (50.0 * b1 / 210.0).floor() + 1.0) * 105.0) as u64;
        bounds.push((b1 as u64, b2));
        b1 += b1.sqrt();
    }
    bounds
}
```

**Step 5: Implement the full cofactorization pipeline**

```rust
//! Cofactorization pipeline: trial div → P-1 → P+1 → ECM.

pub mod trialdiv;
pub mod pm1;
pub mod pp1;
pub mod ecm;

use crate::arith::{is_probable_prime, TrialDivisor};
use crate::params::NfsParams;

#[derive(Debug, Clone)]
pub enum CofactResult {
    /// Fully smooth (cofactor = 1).
    Smooth(Vec<(u32, u8)>),
    /// One large prime on this side.
    OneLargePrime(Vec<(u32, u8)>, u64),
    /// Two large primes on this side.
    TwoLargePrimes(Vec<(u32, u8)>, u64, u64),
    /// Not smooth — discard.
    NotSmooth,
}

/// Full cofactorization of a norm.
pub fn cofactorize(
    norm: u64,
    divisors: &[TrialDivisor],
    lpb: u32,
    mfb: u32,
    lim: u64,
) -> CofactResult {
    let (factors, cofactor) = trialdiv::trial_divide(norm, divisors);

    if cofactor <= 1 {
        return CofactResult::Smooth(factors);
    }

    let l = 1u64 << lpb;
    if cofactor <= l {
        return CofactResult::OneLargePrime(factors, cofactor);
    }

    if (64 - cofactor.leading_zeros()) as u32 > mfb {
        return CofactResult::NotSmooth;
    }

    // Dead-zone check: if cofactor is prime and > l, it can't help
    if is_probable_prime(cofactor) {
        return CofactResult::NotSmooth;
    }

    // P-1 (B1=315, B2=2205)
    if let Some(f) = pm1::pm1(cofactor, 315, 2205) {
        return check_split(factors, f, cofactor / f, l);
    }

    // P+1 (B1=525, B2=3255)
    if let Some(f) = pp1::pp1(cofactor, 525, 3255) {
        return check_split(factors, f, cofactor / f, l);
    }

    // ECM chain
    for (b1, b2) in ecm::ecm_bounds(lpb) {
        if let Some(f) = ecm::ecm_one_curve(cofactor, b1, b2, b1) {
            return check_split(factors, f, cofactor / f, l);
        }
    }

    CofactResult::NotSmooth
}

fn check_split(
    mut factors: Vec<(u32, u8)>,
    f1: u64, f2: u64, l: u64,
) -> CofactResult {
    if f1 <= l && f2 <= l {
        CofactResult::TwoLargePrimes(factors, f1, f2)
    } else {
        CofactResult::NotSmooth
    }
}
```

**Step 6: Write tests for each method individually + the full pipeline.**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pm1_finds_factor() {
        // 1000003 has factor structure amenable to P-1
        // 1000003 = 1000003 (prime), so use a composite
        let n = 1_000_003u64 * 1_000_033;
        // One of these primes p has p-1 = B1-smooth for moderate B1
        let f = pm1::pm1(n, 10_000, 100_000);
        // P-1 may or may not find it depending on smoothness of p-1
        // Just verify it returns a valid factor if it finds one
        if let Some(factor) = f {
            assert!(n % factor == 0);
            assert!(factor > 1 && factor < n);
        }
    }

    #[test]
    fn test_cofactorize_smooth() {
        // 2 * 3 * 5 * 7 = 210
        let divisors: Vec<_> = [2u64, 3, 5, 7, 11, 13]
            .iter()
            .map(|&p| crate::arith::TrialDivisor::new(p, 1.0))
            .collect();
        match cofactorize(210, &divisors, 17, 18, 30000) {
            CofactResult::Smooth(_) => {} // correct
            other => panic!("expected Smooth, got {:?}", other),
        }
    }
}
```

**Step 7: Run tests, verify pass, commit.**

---

## Task 10: Filtering (Singleton + Duplicate Removal)

**Files:**
- Create: `rust/rust-nfs/src/filter.rs`
- Modify: `rust/rust-nfs/src/lib.rs` — add `pub mod filter;`

**Step 1: Implement singleton and duplicate removal**

```rust
//! Filtering: remove duplicate and singleton relations.

use std::collections::{HashMap, HashSet};
use crate::relation::Relation;

/// Remove duplicate (a,b) pairs and relations containing singleton ideals.
pub fn filter_relations(relations: Vec<Relation>) -> Vec<Relation> {
    // Step 1: Deduplicate by (a, b)
    let mut seen = HashSet::new();
    let mut unique: Vec<Relation> = relations
        .into_iter()
        .filter(|r| seen.insert((r.a, r.b)))
        .collect();

    // Step 2: Iterative singleton removal
    loop {
        let mut col_weight: HashMap<u64, u32> = HashMap::new();
        for rel in &unique {
            for &(idx, _) in &rel.rational_factors {
                *col_weight.entry(idx as u64).or_default() += 1;
            }
            for &(idx, _) in &rel.algebraic_factors {
                *col_weight.entry(1_000_000 + idx as u64).or_default() += 1;
            }
            if rel.rat_cofactor > 1 {
                *col_weight.entry(2_000_000 + rel.rat_cofactor).or_default() += 1;
            }
            if rel.alg_cofactor > 1 {
                *col_weight.entry(3_000_000 + rel.alg_cofactor).or_default() += 1;
            }
        }

        let before = unique.len();
        unique.retain(|rel| {
            let has_singleton = rel.rational_factors.iter().any(|&(idx, _)| {
                col_weight.get(&(idx as u64)).copied().unwrap_or(0) < 2
            }) || rel.algebraic_factors.iter().any(|&(idx, _)| {
                col_weight.get(&(1_000_000 + idx as u64)).copied().unwrap_or(0) < 2
            });
            !has_singleton
        });

        if unique.len() == before { break; }
    }

    unique
}
```

**Step 2: Test, commit.**

---

## Task 11: Full Pipeline Integration

**Files:**
- Create: `rust/rust-nfs/src/pipeline.rs`
- Modify: `rust/rust-nfs/src/lib.rs` — add `pub mod pipeline; pub mod filter;`

**Step 1: Implement the full NFS pipeline**

```rust
//! Full NFS pipeline: poly selection → sieve → filter → LA → sqrt.

use rug::Integer;
use crate::params::NfsParams;
use crate::relation::Relation;

#[derive(Debug, Clone, serde::Serialize)]
pub struct NfsResult {
    pub n: String,
    pub factor: Option<String>,
    pub relations_found: usize,
    pub relations_after_filter: usize,
    pub matrix_rows: usize,
    pub matrix_cols: usize,
    pub dependencies_found: usize,
    pub sieve_ms: f64,
    pub filter_ms: f64,
    pub la_ms: f64,
    pub sqrt_ms: f64,
    pub total_ms: f64,
}

/// Factor N using the full NFS pipeline.
pub fn factor_nfs(n: &Integer, params: &NfsParams) -> NfsResult { todo!() }
```

The implementation will:
1. Call `gnfs::polyselect::select_base_m(n, params.degree)` for polynomial
2. Build factor bases for both sides
3. Call `sieve::sieve_specialq()` to collect relations
4. Call `filter::filter_relations()` to clean up
5. Convert relations to `gnfs::types::Relation` via `to_gnfs_relation()`
6. Call `gnfs::linalg::build_matrix()` and `find_dependencies()`
7. Call `gnfs::sqrt::extract_factor_verbose()` for each dependency

**Step 2: Integration test — factor a known 64-bit semiprime end-to-end.**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_factor_small() {
        let n = Integer::from(8051u64); // 83 * 97
        let params = NfsParams::c30();
        let result = factor_nfs(&n, &params);
        assert!(result.factor.is_some(), "should factor 8051");
        let f: Integer = result.factor.unwrap().parse().unwrap();
        assert!(Integer::from(&n % &f) == 0);
        assert!(f > 1 && f < n);
    }
}
```

**Step 3: Implement, test, commit.**

---

## Task 12: CLI + Benchmark + Head-to-Head

**Files:**
- Modify: `rust/rust-nfs/src/main.rs` — full CLI implementation

**Step 1: Implement CLI with factor and benchmark modes**

```rust
fn main() {
    let cli = Cli::parse();

    if let Some(ref n_str) = cli.factor {
        // Factor mode: factor a single number
        let n: Integer = n_str.parse().expect("invalid number");
        let bits = n.significant_bits();
        let params = NfsParams::for_bits(bits as u32);
        eprintln!("Factoring {} ({} bits) with {} params", n, bits, params.name);
        let result = pipeline::factor_nfs(&n, &params);
        println!("{}", serde_json::to_string_pretty(&result).unwrap());
    }

    if let Some(ref bit_sizes) = cli.bits {
        // Benchmark mode: generate semiprimes and factor them
        benchmark(bit_sizes, cli.semiprimes);
    }
}

fn benchmark(bit_sizes: &[u32], semiprimes_per_size: usize) {
    use factoring_core::generate_rsa_target;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    for &bits in bit_sizes {
        let params = NfsParams::for_bits(bits);
        let mut rng = StdRng::seed_from_u64(42 + bits as u64);

        eprintln!("=== {} bits ({}) ===", bits, params.name);
        let mut total_rels = 0usize;
        let mut total_ms = 0.0f64;

        for i in 0..semiprimes_per_size {
            let target = generate_rsa_target(bits, &mut rng);
            let n = &target.n;
            let start = std::time::Instant::now();
            let result = pipeline::factor_nfs(n, &params);
            let elapsed = start.elapsed().as_secs_f64() * 1000.0;

            eprintln!("  [{}/{}] N={:.16}... → {} rels, {:.0}ms, factor={:?}",
                i + 1, semiprimes_per_size, n.to_string(),
                result.relations_found, elapsed, result.factor);

            total_rels += result.relations_found;
            total_ms += elapsed;
        }

        let mean_rps = total_rels as f64 / (total_ms / 1000.0);
        eprintln!("  >> {}-bit: {:.0} rels/sec mean", bits, mean_rps);
    }
}
```

**Step 2: Run benchmark at 96-bit to validate**

Run: `cd /Users/andriipotapov/Semiprime/rust/rust-nfs && cargo run --release -- --bits 96 --semiprimes 1`
Expected: Factors a 96-bit semiprime. Record rels/sec.

**Step 3: Run CADO-NFS on same semiprime for comparison**

Use existing `cado-evolve` wrapper or direct CADO invocation.

**Step 4: Head-to-head at 96, 112, 128 bits**

Run both implementations on same 3 semiprimes at each size (seed=42).
Record: rels/sec, wall time, factor found.

**Step 5: Commit benchmark results**

```bash
git add rust/rust-nfs/
git commit -m "feat(rust-nfs): full NFS pipeline with benchmark harness"
```

---

## Summary: Implementation Order and Dependencies

```
Task 1 (scaffold) ← no deps
  ↓
Task 2 (Montgomery arith) ← Task 1
  ↓
Task 3 (factor base) ← Task 2
  ↓
Task 4 (q-lattice) ← Task 1
  ↓
Task 5 (bucket sieve) ← Task 1
  ↓
Task 6 (norm + small sieve) ← Task 3
  ↓
Task 7 (per-region) ← Task 5, Task 6
  ↓
Task 8 (special-q loop) ← Task 4, Task 7
  ↓
Task 9 (cofactorization) ← Task 2
  ↓
Task 10 (filtering) ← Task 1
  ↓
Task 11 (pipeline) ← Task 8, Task 9, Task 10
  ↓
Task 12 (CLI + benchmark) ← Task 11
```

**Parallelizable pairs**: Tasks 4+5 can run in parallel. Tasks 9+10 can run in parallel.

**Critical path**: 1 → 2 → 3 → 6 → 7 → 8 → 11 → 12

**Estimated tasks**: 12 tasks, ~5-8 TDD steps each.
