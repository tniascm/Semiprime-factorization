use rug::ops::Pow;
use rug::Integer;
use serde::{Deserialize, Serialize};

/// A polynomial pair (f, g) where f(m) ≡ g(m) ≡ 0 (mod N).
/// g(x) = g0 + g1*x is linear (rational side).
/// f(x) = c0 + c1*x + ... + cd*x^d is the algebraic polynomial.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolynomialPair {
    pub f_coeffs_str: Vec<String>,
    pub g_coeffs_str: Vec<String>,
    pub m_str: String,
    pub degree: u32,
    pub n_str: String,
}

impl PolynomialPair {
    pub fn new(f_coeffs: &[Integer], g0: &Integer, g1: &Integer, m: &Integer, n: &Integer) -> Self {
        Self {
            f_coeffs_str: f_coeffs.iter().map(|c| c.to_string()).collect(),
            g_coeffs_str: vec![g0.to_string(), g1.to_string()],
            m_str: m.to_string(),
            degree: (f_coeffs.len() - 1) as u32,
            n_str: n.to_string(),
        }
    }

    pub fn f_coeffs(&self) -> Vec<Integer> {
        self.f_coeffs_str
            .iter()
            .map(|s| s.parse::<Integer>().unwrap())
            .collect()
    }

    pub fn g0(&self) -> Integer {
        self.g_coeffs_str[0].parse().unwrap()
    }
    pub fn g1(&self) -> Integer {
        self.g_coeffs_str[1].parse().unwrap()
    }
    pub fn m(&self) -> Integer {
        self.m_str.parse().unwrap()
    }
    pub fn n(&self) -> Integer {
        self.n_str.parse().unwrap()
    }

    /// Evaluate f(a, b) = b^d * f(a/b) = c0*b^d + c1*a*b^(d-1) + ... + cd*a^d
    pub fn eval_f_homogeneous(&self, a: i64, b: u64) -> Integer {
        let coeffs = self.f_coeffs();
        let d = coeffs.len() - 1;
        let a_int = Integer::from(a);
        let b_int = Integer::from(b);
        let mut result = Integer::from(0);
        for (i, c) in coeffs.iter().enumerate() {
            let a_pow = a_int.clone().pow(i as u32);
            let b_pow = b_int.clone().pow((d - i) as u32);
            let term = Integer::from(c * &a_pow) * &b_pow;
            result += term;
        }
        result
    }

    /// Evaluate g(a, b) = a - b*m (rational norm, from g(x) = x - m).
    pub fn eval_g(&self, a: i64, b: u64) -> Integer {
        let m = self.m();
        Integer::from(a) - Integer::from(b) * m
    }
}

/// A smooth relation: (a, b) pair with factorization on both sides.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Relation {
    pub a: i64,
    pub b: u64,
    pub rational_factors: Vec<(u32, u8)>,
    pub algebraic_factors: Vec<(u32, u8)>,
    pub rational_sign_negative: bool,
    pub algebraic_sign_negative: bool,
    /// Special-q prime and root (q, r) if this relation came from a lattice sieve.
    /// None for line-sieve relations.
    pub special_q: Option<(u64, u64)>,
    /// Single large prime on rational side (prime not in FB, appears with exponent 1).
    pub rat_lp: Option<u64>,
    /// Single large prime on algebraic side: (prime, root) identifying the ideal.
    pub alg_lp: Option<(u64, u64)>,
}

/// Factor base: sorted list of primes with precomputed roots.
#[derive(Debug, Clone)]
pub struct FactorBase {
    pub primes: Vec<u64>,
    pub algebraic_roots: Vec<Vec<u64>>,
    pub log_p: Vec<u8>,
}

impl FactorBase {
    /// Total number of (prime, root) pairs across all factor base primes.
    /// Each pair corresponds to a degree-1 prime ideal above that prime.
    pub fn algebraic_pair_count(&self) -> usize {
        self.algebraic_roots.iter().map(|r| r.len()).sum()
    }

    /// Flat index offset for prime at index `prime_idx`.
    /// The (prime_idx, root_idx) pair has flat index = pair_offset(prime_idx) + root_idx.
    pub fn pair_offset(&self, prime_idx: usize) -> usize {
        self.algebraic_roots[..prime_idx]
            .iter()
            .map(|r| r.len())
            .sum()
    }

    /// Count of primes that have a higher-degree ideal factor.
    /// For a degree-d polynomial, primes with k < d roots have a residual ideal
    /// of degree (d - k). We need one matrix column per such prime.
    pub fn higher_degree_ideal_count(&self, poly_degree: usize) -> usize {
        self.algebraic_roots
            .iter()
            .filter(|roots| !roots.is_empty() && roots.len() < poly_degree)
            .count()
    }

    /// Flat offset for the higher-degree ideal column of prime at `prime_idx`.
    /// Returns None if this prime has no HD ideal (fully splits or no roots).
    pub fn hd_offset(&self, prime_idx: usize, poly_degree: usize) -> Option<usize> {
        let roots = &self.algebraic_roots[prime_idx];
        if roots.is_empty() || roots.len() >= poly_degree {
            return None;
        }
        let offset = self.algebraic_roots[..prime_idx]
            .iter()
            .filter(|r| !r.is_empty() && r.len() < poly_degree)
            .count();
        Some(offset)
    }
}

/// Sparse GF(2) matrix row, stored as a bitset.
#[derive(Debug, Clone)]
pub struct BitRow {
    pub bits: Vec<u64>,
    pub ncols: usize,
}

impl BitRow {
    pub fn new(ncols: usize) -> Self {
        let nwords = (ncols + 63) / 64;
        Self {
            bits: vec![0u64; nwords],
            ncols,
        }
    }

    pub fn get(&self, col: usize) -> bool {
        let word = col / 64;
        let bit = col % 64;
        (self.bits[word] >> bit) & 1 == 1
    }

    pub fn set(&mut self, col: usize) {
        let word = col / 64;
        let bit = col % 64;
        self.bits[word] |= 1u64 << bit;
    }

    pub fn flip(&mut self, col: usize) {
        let word = col / 64;
        let bit = col % 64;
        self.bits[word] ^= 1u64 << bit;
    }

    pub fn xor_with(&mut self, other: &BitRow) {
        let len = self.bits.len().min(other.bits.len());
        let a = &mut self.bits[..len];
        let b = &other.bits[..len];
        xor_slice(a, b);
    }

    pub fn is_zero(&self) -> bool {
        self.bits.iter().all(|&w| w == 0)
    }
}

/// XOR `src` into `dst` element-wise. Uses NEON on aarch64.
#[inline]
fn xor_slice(dst: &mut [u64], src: &[u64]) {
    #[cfg(target_arch = "aarch64")]
    {
        xor_slice_neon(dst, src);
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        for (a, b) in dst.iter_mut().zip(src.iter()) {
            *a ^= *b;
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
fn xor_slice_neon(dst: &mut [u64], src: &[u64]) {
    use core::arch::aarch64::*;
    let len = dst.len().min(src.len());

    // Process 4 u64s (256 bits) at a time using pairs of 128-bit NEON ops
    let chunks4 = len / 4;
    let dp = dst.as_mut_ptr();
    let sp = src.as_ptr();
    unsafe {
        for i in 0..chunks4 {
            let off = i * 4;
            let a0 = vld1q_u64(dp.add(off));
            let b0 = vld1q_u64(sp.add(off));
            let a1 = vld1q_u64(dp.add(off + 2));
            let b1 = vld1q_u64(sp.add(off + 2));
            vst1q_u64(dp.add(off), veorq_u64(a0, b0));
            vst1q_u64(dp.add(off + 2), veorq_u64(a1, b1));
        }
    }

    // Handle remaining elements
    for i in (chunks4 * 4)..len {
        dst[i] ^= src[i];
    }
}
