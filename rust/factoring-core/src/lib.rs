//! Shared types, utilities, and RSA key generation for factorization experiments.

use num_bigint::BigUint;
use num_integer::Integer;
use num_traits::{One, Zero};
use rand::Rng;
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// Result of a factorization attempt.
#[derive(Debug, Clone)]
pub struct FactorResult {
    /// The number that was factored
    pub n: BigUint,
    /// Found factors (empty if factorization failed)
    pub factors: Vec<BigUint>,
    /// Which algorithm produced this result
    pub algorithm: Algorithm,
    /// Time taken
    pub duration: Duration,
    /// Whether the factorization was complete (all factors found)
    pub complete: bool,
}

/// Available factorization algorithms.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Algorithm {
    TrialDivision,
    PollardRho,
    PollardRhoBrent,
    PollardPMinus1,
    Ensemble,
    ECM,
    QuadraticSieve,
    NumberFieldSieve,
    LatticeReduction,
    IsingAnnealing,
    QuantumInspired,
    CompressionGuided,
    GroupStructure,
    MLAGuided,
    MultiBaseAnalysis,
    SmoothPilatte,
    MurruSalvatori,
}

impl fmt::Display for Algorithm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Algorithm::TrialDivision => write!(f, "Trial Division"),
            Algorithm::PollardRho => write!(f, "Pollard's rho"),
            Algorithm::PollardRhoBrent => write!(f, "Pollard's rho (Brent)"),
            Algorithm::PollardPMinus1 => write!(f, "Pollard's p-1"),
            Algorithm::Ensemble => write!(f, "Ensemble"),
            Algorithm::ECM => write!(f, "ECM (Lenstra)"),
            Algorithm::QuadraticSieve => write!(f, "Quadratic Sieve"),
            Algorithm::NumberFieldSieve => write!(f, "Number Field Sieve"),
            Algorithm::LatticeReduction => write!(f, "Lattice Reduction (SVP)"),
            Algorithm::IsingAnnealing => write!(f, "Ising/QUBO Annealing"),
            Algorithm::QuantumInspired => write!(f, "Quantum-Inspired"),
            Algorithm::CompressionGuided => write!(f, "Compression-Guided"),
            Algorithm::GroupStructure => write!(f, "Group Structure Analysis"),
            Algorithm::MLAGuided => write!(f, "MLA-Guided"),
            Algorithm::MultiBaseAnalysis => write!(f, "Multi-Base Analysis"),
            Algorithm::SmoothPilatte => write!(f, "Smooth Pilatte (Lattice-Geometric)"),
            Algorithm::MurruSalvatori => write!(f, "Murru-Salvatori (CF + BSGS)"),
        }
    }
}

/// RSA test target — a semiprime with known factors for verification.
#[derive(Debug, Clone)]
pub struct RsaTarget {
    pub n: BigUint,
    pub p: BigUint,
    pub q: BigUint,
    pub bit_size: u32,
}

impl RsaTarget {
    /// Verify that a factorization result is correct.
    pub fn verify(&self, result: &FactorResult) -> bool {
        if result.factors.len() < 2 {
            return false;
        }
        let product: BigUint = result.factors.iter().fold(BigUint::one(), |acc, f| acc * f);
        product == self.n
    }
}

/// Generate a random prime of approximately `bits` bit size.
/// Uses probabilistic primality testing.
pub fn random_prime(bits: u32, rng: &mut impl Rng) -> BigUint {
    assert!(bits >= 2, "Cannot generate a prime with fewer than 2 bits");
    loop {
        let num_bytes = (bits as usize + 7) / 8;
        let mut bytes = vec![0u8; num_bytes];
        rng.fill(&mut bytes[..]);

        // Clear excess high bits to ensure the number fits in `bits` bits.
        // For example, if bits=50 and num_bytes=7 (56 bits), we need to
        // clear the top 6 bits of the first byte.
        let excess_bits = (num_bytes * 8) as u32 - bits;
        if excess_bits > 0 {
            bytes[0] &= (1u8 << (8 - excess_bits)) - 1;
        }

        // Set the top bit (bit `bits-1`) to ensure the number has exactly
        // `bits` bits. The top bit position within the first byte depends
        // on the alignment.
        let top_bit_in_byte = (bits - 1) % 8;
        bytes[0] |= 1u8 << top_bit_in_byte;

        // Set the bottom bit to ensure odd
        if let Some(last) = bytes.last_mut() {
            *last |= 0x01;
        }

        let candidate = BigUint::from_bytes_be(&bytes);
        debug_assert!(
            candidate.bits() == bits as u64,
            "Generated number has {} bits, expected {}",
            candidate.bits(),
            bits
        );
        if is_probably_prime(&candidate, 20) {
            return candidate;
        }
    }
}

/// Generate an RSA test target with the given bit size.
pub fn generate_rsa_target(bits: u32, rng: &mut impl Rng) -> RsaTarget {
    let half_bits = bits / 2;
    let p = random_prime(half_bits, rng);
    let q = random_prime(half_bits, rng);
    let n = &p * &q;
    RsaTarget {
        n,
        p,
        q,
        bit_size: bits,
    }
}

/// Miller-Rabin probabilistic primality test.
pub fn is_probably_prime(n: &BigUint, rounds: u32) -> bool {
    let one = BigUint::one();
    let two = &one + &one;
    let three = &two + &one;

    if *n < two {
        return false;
    }
    if *n == two || *n == three {
        return true;
    }
    if n.is_even() {
        return false;
    }

    // Write n-1 as 2^r * d
    let n_minus_1 = n - &one;
    let mut d = n_minus_1.clone();
    let mut r: u32 = 0;
    while d.is_even() {
        d >>= 1u32;
        r += 1;
    }

    let mut rng = rand::thread_rng();

    'witness: for _ in 0..rounds {
        // Random a in [2, n-2]
        let a = loop {
            let bytes = n.to_bytes_be();
            let mut random_bytes = vec![0u8; bytes.len()];
            rng.fill(&mut random_bytes[..]);
            let a = BigUint::from_bytes_be(&random_bytes) % n;
            if a >= two && a <= &n_minus_1 - &one {
                break a;
            }
        };

        let mut x = mod_pow(&a, &d, n);

        if x == one || x == n_minus_1 {
            continue 'witness;
        }

        for _ in 0..r - 1 {
            x = mod_pow(&x, &two, n);
            if x == n_minus_1 {
                continue 'witness;
            }
        }

        return false;
    }

    true
}

/// Modular exponentiation: base^exp mod modulus.
pub fn mod_pow(base: &BigUint, exp: &BigUint, modulus: &BigUint) -> BigUint {
    base.modpow(exp, modulus)
}

/// Greatest common divisor.
pub fn gcd(a: &BigUint, b: &BigUint) -> BigUint {
    a.gcd(b)
}

/// Trial division up to a given bound.
pub fn trial_division(n: &BigUint, bound: u64) -> Vec<BigUint> {
    let mut factors = Vec::new();
    let mut remaining = n.clone();
    let two = BigUint::from(2u32);

    while remaining.is_even() {
        factors.push(two.clone());
        remaining >>= 1u32;
    }

    let mut divisor = 3u64;
    while divisor <= bound && BigUint::from(divisor) * BigUint::from(divisor) <= remaining {
        let big_divisor = BigUint::from(divisor);
        while (&remaining % &big_divisor).is_zero() {
            factors.push(big_divisor.clone());
            remaining /= &big_divisor;
        }
        divisor += 2;
    }

    if remaining > BigUint::one() {
        factors.push(remaining);
    }

    factors
}

/// Pollard's rho algorithm for factorization.
///
/// Retries with different random parameters up to `max_attempts` times
/// to handle the probabilistic nature of the algorithm.
pub fn pollard_rho(n: &BigUint) -> Option<BigUint> {
    pollard_rho_with_attempts(n, 20)
}

/// Pollard's rho with a configurable number of retry attempts.
fn pollard_rho_with_attempts(n: &BigUint, max_attempts: u32) -> Option<BigUint> {
    let one = BigUint::one();
    let two = BigUint::from(2u32);

    if n.is_even() {
        return Some(two);
    }

    let mut rng = rand::thread_rng();

    for _ in 0..max_attempts {
        let c = loop {
            let bytes = n.to_bytes_be();
            let mut random_bytes = vec![0u8; bytes.len()];
            rng.fill(&mut random_bytes[..]);
            let c = BigUint::from_bytes_be(&random_bytes) % n;
            if !c.is_zero() && c != n - &two {
                break c;
            }
        };

        let f = |x: &BigUint| -> BigUint { (x * x + &c) % n };

        let mut x = BigUint::from(2u32);
        let mut y = BigUint::from(2u32);

        let mut found_trivial = false;
        for _ in 0..1_000_000 {
            x = f(&x);
            y = f(&f(&y));

            let diff = if x > y { &x - &y } else { &y - &x };
            let d = gcd(&diff, n);

            if d == one {
                continue;
            }
            if d == *n {
                found_trivial = true;
                break;
            }
            return Some(d);
        }

        if found_trivial {
            // Retry with a different c value
            continue;
        }
    }

    None
}

/// Represent a number in an arbitrary base.
pub fn to_base(n: &BigUint, base: u32) -> Vec<u32> {
    if n.is_zero() {
        return vec![0];
    }

    let base_big = BigUint::from(base);
    let mut digits = Vec::new();
    let mut remaining = n.clone();

    while !remaining.is_zero() {
        let digit = &remaining % &base_big;
        digits.push(digit.to_u32_digits().first().copied().unwrap_or(0));
        remaining /= &base_big;
    }

    digits.reverse();
    digits
}

/// Compute the Residue Number System representation.
/// Returns (N mod m1, N mod m2, ..., N mod mk) for given moduli.
pub fn to_rns(n: &BigUint, moduli: &[u64]) -> Vec<u64> {
    moduli
        .iter()
        .map(|&m| {
            let m_big = BigUint::from(m);
            let remainder = n % &m_big;
            remainder.to_u64_digits().first().copied().unwrap_or(0)
        })
        .collect()
}

/// Shannon entropy of a digit sequence.
pub fn entropy(digits: &[u32], base: u32) -> f64 {
    let total = digits.len() as f64;
    if total == 0.0 {
        return 0.0;
    }

    let mut counts = vec![0u64; base as usize];
    for &d in digits {
        if (d as usize) < counts.len() {
            counts[d as usize] += 1;
        }
    }

    counts
        .iter()
        .filter(|&&c| c > 0)
        .map(|&c| {
            let p = c as f64 / total;
            -p * p.log2()
        })
        .sum()
}

use num_traits::ToPrimitive;

/// Convert BigUint to u64 (panics if too large).
pub fn to_u64(n: &BigUint) -> u64 {
    n.to_u64().expect("number too large for u64")
}

/// Generate all primes up to `limit` using the Sieve of Eratosthenes.
fn sieve_primes(limit: u64) -> Vec<u64> {
    if limit < 2 {
        return Vec::new();
    }
    let size = (limit + 1) as usize;
    let mut is_prime = vec![true; size];
    is_prime[0] = false;
    is_prime[1] = false;
    let mut i = 2usize;
    while i * i < size {
        if is_prime[i] {
            let mut j = i * i;
            while j < size {
                is_prime[j] = false;
                j += i;
            }
        }
        i += 1;
    }
    is_prime
        .iter()
        .enumerate()
        .filter(|(_, &p)| p)
        .map(|(i, _)| i as u64)
        .collect()
}

/// Pollard's p-1 factorization algorithm.
///
/// Exploits primes p where p-1 is B-smooth (all prime factors of p-1 are <= `bound`).
/// Starts with a = 2 and raises it to successive prime powers <= bound modulo n.
/// If gcd(a-1, n) yields a nontrivial factor, returns it.
pub fn pollard_p_minus_1(n: &BigUint, bound: u64) -> Option<BigUint> {
    let one = BigUint::one();

    if n <= &one {
        return None;
    }
    if n.is_even() {
        return Some(BigUint::from(2u32));
    }

    let primes = sieve_primes(bound);
    let mut a = BigUint::from(2u32);

    for &p in &primes {
        // Compute the largest power of p that is <= bound: p^k where p^k <= bound
        let mut pk = p;
        while pk <= bound / p {
            pk *= p;
        }
        // a = a^pk mod n
        a = mod_pow(&a, &BigUint::from(pk), n);

        // Check gcd(a - 1, n)
        if a <= one {
            // a became 0 or 1 mod n, this attempt won't work with this base
            return None;
        }
        let a_minus_1 = &a - &one;
        let d = gcd(&a_minus_1, n);
        if d > one && d < *n {
            return Some(d);
        }
        if d == *n {
            // gcd is n itself, the bound may be too large or unlucky
            return None;
        }
    }

    None
}

/// Unified automatic factorization dispatcher.
///
/// Strategy based on bit size:
/// - <= 32 bits: trial division up to sqrt(n)
/// - <= 64 bits: try Pollard rho, then Pollard p-1 (bound=10000)
/// - > 64 bits: try Pollard p-1 (bound=100000), then Pollard rho
///
/// Returns a `FactorResult` with the successful algorithm, found factors (including
/// the cofactor n/factor), and timing information.
pub fn factor_auto(n: &BigUint) -> FactorResult {
    let start = std::time::Instant::now();
    let one = BigUint::one();
    let bits = n.bits();

    // Handle trivial cases
    if *n <= one {
        return FactorResult {
            n: n.clone(),
            factors: vec![n.clone()],
            algorithm: Algorithm::TrialDivision,
            duration: start.elapsed(),
            complete: true,
        };
    }

    if bits <= 32 {
        // Small numbers: trial division up to sqrt(n)
        let bound = to_u64(n).isqrt();
        let factors = trial_division(n, bound);
        return FactorResult {
            n: n.clone(),
            factors,
            algorithm: Algorithm::TrialDivision,
            duration: start.elapsed(),
            complete: true,
        };
    }

    if bits <= 64 {
        // Medium numbers: try Pollard rho first, then p-1
        if let Some(factor) = pollard_rho(n) {
            let cofactor = n / &factor;
            let mut factors = vec![factor, cofactor];
            factors.sort();
            return FactorResult {
                n: n.clone(),
                factors,
                algorithm: Algorithm::PollardRho,
                duration: start.elapsed(),
                complete: true,
            };
        }
        if let Some(factor) = pollard_p_minus_1(n, 10_000) {
            let cofactor = n / &factor;
            let mut factors = vec![factor, cofactor];
            factors.sort();
            return FactorResult {
                n: n.clone(),
                factors,
                algorithm: Algorithm::PollardPMinus1,
                duration: start.elapsed(),
                complete: true,
            };
        }
    } else {
        // Large numbers: try p-1 first (good for smooth factors), then rho
        if let Some(factor) = pollard_p_minus_1(n, 100_000) {
            let cofactor = n / &factor;
            let mut factors = vec![factor, cofactor];
            factors.sort();
            return FactorResult {
                n: n.clone(),
                factors,
                algorithm: Algorithm::PollardPMinus1,
                duration: start.elapsed(),
                complete: true,
            };
        }
        if let Some(factor) = pollard_rho(n) {
            let cofactor = n / &factor;
            let mut factors = vec![factor, cofactor];
            factors.sort();
            return FactorResult {
                n: n.clone(),
                factors,
                algorithm: Algorithm::PollardRho,
                duration: start.elapsed(),
                complete: true,
            };
        }
    }

    // All methods failed
    FactorResult {
        n: n.clone(),
        factors: vec![],
        algorithm: Algorithm::TrialDivision,
        duration: start.elapsed(),
        complete: false,
    }
}

/// Compute the modular multiplicative inverse: a^(-1) mod modulus.
///
/// Uses the extended Euclidean algorithm internally with signed arithmetic
/// (tracking signs manually since BigUint is unsigned).
/// Returns `None` if gcd(a, modulus) != 1 (inverse does not exist).
pub fn mod_inverse(a: &BigUint, modulus: &BigUint) -> Option<BigUint> {
    _mod_inverse_impl(a, modulus)
}

/// Internal helper: compute modular inverse using a clean iterative extended GCD.
/// Returns a^(-1) mod m, or None if gcd(a, m) != 1.
fn _mod_inverse_impl(a: &BigUint, m: &BigUint) -> Option<BigUint> {
    let one = BigUint::one();
    if m <= &one {
        return None;
    }

    let a_mod = a % m;
    if a_mod.is_zero() {
        return None;
    }

    // We use the iterative extended Euclidean algorithm.
    // Maintain: old_r = a, r = m and track coefficient for a only.
    // At each step: old_r = q * r + remainder
    //
    // Since BigUint cannot represent negative numbers, we track signs manually.

    let mut old_r = a_mod;
    let mut r = m.clone();

    // Coefficients: old_s * a_original + old_t * m = old_r
    // We only track old_s (coefficient of a).
    let mut old_s = BigUint::one();
    let mut old_s_neg = false;
    let mut s_val = BigUint::zero();
    let mut s_neg = false;

    while !r.is_zero() {
        let q = &old_r / &r;

        // Update r
        let new_r = &old_r % &r;
        old_r = r;
        r = new_r;

        // Update s: new_s = old_s - q * s
        let q_times_s = &q * &s_val;

        // Compute old_s - q_times_s with sign tracking
        // old_s has sign old_s_neg, q_times_s has sign s_neg
        let (new_s, new_s_neg) = if old_s_neg == s_neg {
            // Both same sign: result = |old_s| - |q*s|, keep sign if old_s >= q*s
            if old_s >= q_times_s {
                (&old_s - &q_times_s, old_s_neg)
            } else {
                (&q_times_s - &old_s, !old_s_neg)
            }
        } else {
            // Different signs: old_s - (-q_times_s) = old_s + q_times_s
            (&old_s + &q_times_s, old_s_neg)
        };

        old_s = s_val;
        old_s_neg = s_neg;
        s_val = new_s;
        s_neg = if s_val.is_zero() { false } else { new_s_neg };
    }

    // old_r should be gcd(a, m)
    if old_r != one {
        return None;
    }

    // old_s * a ≡ 1 (mod m)
    if old_s_neg {
        // Negative: add m to make positive
        Some(m - (&old_s % m))
    } else {
        Some(&old_s % m)
    }
}

/// Pollard's rho algorithm with Brent's cycle detection improvement.
///
/// Brent's variant uses a power-of-2 cycle detection strategy that is faster
/// than Floyd's tortoise-and-hare in practice. It also accumulates products
/// and performs batch GCD every ~100 steps, reducing the number of expensive
/// GCD operations.
///
/// Returns a nontrivial factor of `n`, or `None` if the algorithm fails.
pub fn pollard_rho_brent(n: &BigUint) -> Option<BigUint> {
    pollard_rho_brent_with_attempts(n, 20)
}

/// Pollard's rho (Brent variant) with configurable retry attempts.
fn pollard_rho_brent_with_attempts(n: &BigUint, max_attempts: u32) -> Option<BigUint> {
    let one = BigUint::one();
    let two = BigUint::from(2u32);

    if n.is_even() {
        return Some(two);
    }
    if *n <= one {
        return None;
    }

    let mut rng = rand::thread_rng();

    for _ in 0..max_attempts {
        // Pick random c in [1, n-1]
        let c = loop {
            let bytes = n.to_bytes_be();
            let mut random_bytes = vec![0u8; bytes.len()];
            rng.fill(&mut random_bytes[..]);
            let c = BigUint::from_bytes_be(&random_bytes) % n;
            if !c.is_zero() && c != n - &two {
                break c;
            }
        };

        // Pick random starting y in [1, n-1]
        let mut y = loop {
            let bytes = n.to_bytes_be();
            let mut random_bytes = vec![0u8; bytes.len()];
            rng.fill(&mut random_bytes[..]);
            let val = BigUint::from_bytes_be(&random_bytes) % n;
            if !val.is_zero() {
                break val;
            }
        };

        let f = |x: &BigUint| -> BigUint { (x * x + &c) % n };

        let mut r: u64 = 1; // current power of 2
        let mut q = BigUint::one(); // accumulated product for batch GCD
        let mut ys = y.clone(); // saved y for backtracking
        let mut x = y.clone(); // the "fixed" point at the start of each power-of-2 block
        let mut d = BigUint::one();

        let mut found_trivial = false;

        'outer: while d == one {
            x = y.clone();
            // Advance y by r steps (the tortoise doesn't move; x stays fixed)
            for _ in 0..r {
                y = f(&y);
            }

            // Now look for a factor in chunks within this power-of-2 block
            let mut k: u64 = 0;
            while k < r && d == one {
                ys = y.clone();
                let batch_size = std::cmp::min(100, r - k);
                for _ in 0..batch_size {
                    y = f(&y);
                    let diff = if y > x { &y - &x } else { &x - &y };
                    q = (q * &diff) % n;
                }
                d = gcd(&q, n);
                k += batch_size;
            }

            r *= 2;

            // Safety limit to avoid infinite loops
            if r > 2_000_000 {
                found_trivial = true;
                break 'outer;
            }
        }

        if d == *n || found_trivial {
            // Batch GCD gave trivial result; backtrack from ys and do step-by-step GCD
            let d2 = loop {
                ys = f(&ys);
                let diff = if ys > x { &ys - &x } else { &x - &ys };
                let g = gcd(&diff, n);
                if g != one {
                    break g;
                }
            };
            if d2 == *n {
                // Truly trivial, retry with different c
                continue;
            }
            return Some(d2);
        }

        if d > one && d < *n {
            return Some(d);
        }
    }

    None
}

/// Pollard's rho (Brent) with an early-stop flag for ensemble use.
///
/// Checks `stop_flag` periodically and returns `None` if another thread
/// has already found a factor.
fn pollard_rho_brent_stoppable(
    n: &BigUint,
    stop_flag: &AtomicBool,
    seed: u64,
) -> Option<BigUint> {
    let one = BigUint::one();
    let two = BigUint::from(2u32);

    if n.is_even() {
        return Some(two);
    }
    if *n <= one {
        return None;
    }

    // Use a deterministic-ish seed derived from the seed parameter
    let c = {
        let seed_big = BigUint::from(seed) % n;
        if seed_big.is_zero() || seed_big == n - &two {
            BigUint::from(seed + 1) % n
        } else {
            seed_big
        }
    };

    let mut y = BigUint::from(seed.wrapping_mul(7).wrapping_add(3)) % n;
    if y.is_zero() {
        y = one.clone();
    }

    let f = |x: &BigUint| -> BigUint { (x * x + &c) % n };

    let mut r: u64 = 1;
    let mut q = BigUint::one();
    let mut ys = y.clone();
    let mut x = y.clone();
    let mut d = BigUint::one();

    while d == one {
        if stop_flag.load(Ordering::Relaxed) {
            return None;
        }

        x = y.clone();
        for _ in 0..r {
            y = f(&y);
        }

        let mut k: u64 = 0;
        while k < r && d == one {
            if stop_flag.load(Ordering::Relaxed) {
                return None;
            }

            ys = y.clone();
            let batch_size = std::cmp::min(100, r - k);
            for _ in 0..batch_size {
                y = f(&y);
                let diff = if y > x { &y - &x } else { &x - &y };
                q = (q * &diff) % n;
            }
            d = gcd(&q, n);
            k += batch_size;
        }

        r *= 2;

        if r > 2_000_000 {
            return None;
        }
    }

    if d == *n {
        let d2 = loop {
            if stop_flag.load(Ordering::Relaxed) {
                return None;
            }
            ys = f(&ys);
            let diff = if ys > x { &ys - &x } else { &x - &ys };
            let g = gcd(&diff, n);
            if g != one {
                break g;
            }
        };
        if d2 == *n {
            return None;
        }
        return Some(d2);
    }

    if d > one && d < *n {
        Some(d)
    } else {
        None
    }
}

/// Pollard's p-1 with an early-stop flag for ensemble use.
fn pollard_p_minus_1_stoppable(
    n: &BigUint,
    bound: u64,
    stop_flag: &AtomicBool,
) -> Option<BigUint> {
    let one = BigUint::one();

    if n <= &one {
        return None;
    }
    if n.is_even() {
        return Some(BigUint::from(2u32));
    }

    let primes = sieve_primes(bound);
    let mut a = BigUint::from(2u32);

    for (i, &p) in primes.iter().enumerate() {
        // Check stop flag every 50 primes
        if i % 50 == 0 && stop_flag.load(Ordering::Relaxed) {
            return None;
        }

        let mut pk = p;
        while pk <= bound / p {
            pk *= p;
        }
        a = mod_pow(&a, &BigUint::from(pk), n);

        if a <= one {
            return None;
        }
        let a_minus_1 = &a - &one;
        let d = gcd(&a_minus_1, n);
        if d > one && d < *n {
            return Some(d);
        }
        if d == *n {
            return None;
        }
    }

    None
}

/// An individual ensemble task result, sent from worker threads.
struct EnsembleHit {
    factor: BigUint,
    algorithm: Algorithm,
    duration: Duration,
}

/// Parallel ensemble factoring: runs multiple methods concurrently.
///
/// Launches several factoring strategies in parallel using `std::thread::scope`:
///   a) Pollard rho (Brent) with multiple random seeds
///   b) Pollard p-1 with bound 10,000
///   c) Pollard p-1 with bound 100,000
///   d) Trial division up to 1,000,000
///
/// An `AtomicBool` "found" flag is shared among all threads. Once any method
/// discovers a nontrivial factor, the flag is set so that other methods can
/// detect it and stop early.
///
/// Returns a `FactorResult` from the first successful method, including timing.
/// If no method succeeds within `timeout_ms` milliseconds, returns an incomplete result.
pub fn factor_ensemble(n: &BigUint, timeout_ms: u64) -> FactorResult {
    let start = Instant::now();
    let one = BigUint::one();
    let timeout = Duration::from_millis(timeout_ms);

    // Handle trivial cases
    if *n <= one {
        return FactorResult {
            n: n.clone(),
            factors: vec![n.clone()],
            algorithm: Algorithm::Ensemble,
            duration: start.elapsed(),
            complete: true,
        };
    }
    if n.is_even() {
        let two = BigUint::from(2u32);
        let cofactor = n / &two;
        return FactorResult {
            n: n.clone(),
            factors: vec![two, cofactor],
            algorithm: Algorithm::Ensemble,
            duration: start.elapsed(),
            complete: true,
        };
    }

    let found_flag = AtomicBool::new(false);
    let result: Mutex<Option<EnsembleHit>> = Mutex::new(None);

    std::thread::scope(|s| {
        // (a) Pollard rho (Brent) with multiple seeds
        let rho_seeds: Vec<u64> = vec![2, 7, 13, 31, 97, 211];
        for seed in rho_seeds {
            let found_flag_ref = &found_flag;
            let result_ref = &result;
            let n_clone = n.clone();
            let thread_start = start;
            s.spawn(move || {
                if found_flag_ref.load(Ordering::Relaxed) {
                    return;
                }
                if let Some(factor) = pollard_rho_brent_stoppable(&n_clone, found_flag_ref, seed) {
                    let elapsed = thread_start.elapsed();
                    found_flag_ref.store(true, Ordering::Relaxed);
                    let mut guard = result_ref.lock().unwrap();
                    if guard.is_none() {
                        *guard = Some(EnsembleHit {
                            factor,
                            algorithm: Algorithm::PollardRhoBrent,
                            duration: elapsed,
                        });
                    }
                }
            });
        }

        // (b) Pollard p-1 with bound 10,000
        {
            let found_flag_ref = &found_flag;
            let result_ref = &result;
            let n_clone = n.clone();
            let thread_start = start;
            s.spawn(move || {
                if found_flag_ref.load(Ordering::Relaxed) {
                    return;
                }
                if let Some(factor) =
                    pollard_p_minus_1_stoppable(&n_clone, 10_000, found_flag_ref)
                {
                    let elapsed = thread_start.elapsed();
                    found_flag_ref.store(true, Ordering::Relaxed);
                    let mut guard = result_ref.lock().unwrap();
                    if guard.is_none() {
                        *guard = Some(EnsembleHit {
                            factor,
                            algorithm: Algorithm::PollardPMinus1,
                            duration: elapsed,
                        });
                    }
                }
            });
        }

        // (c) Pollard p-1 with bound 100,000
        {
            let found_flag_ref = &found_flag;
            let result_ref = &result;
            let n_clone = n.clone();
            let thread_start = start;
            s.spawn(move || {
                if found_flag_ref.load(Ordering::Relaxed) {
                    return;
                }
                if let Some(factor) =
                    pollard_p_minus_1_stoppable(&n_clone, 100_000, found_flag_ref)
                {
                    let elapsed = thread_start.elapsed();
                    found_flag_ref.store(true, Ordering::Relaxed);
                    let mut guard = result_ref.lock().unwrap();
                    if guard.is_none() {
                        *guard = Some(EnsembleHit {
                            factor,
                            algorithm: Algorithm::PollardPMinus1,
                            duration: elapsed,
                        });
                    }
                }
            });
        }

        // (d) Trial division up to 1,000,000
        {
            let found_flag_ref = &found_flag;
            let result_ref = &result;
            let n_clone = n.clone();
            let thread_start = start;
            s.spawn(move || {
                if found_flag_ref.load(Ordering::Relaxed) {
                    return;
                }
                let factors = trial_division(&n_clone, 1_000_000);
                // trial_division always returns factors; check if it found a nontrivial split
                if factors.len() >= 2 {
                    // The first factor is a genuine small prime factor
                    let factor = factors[0].clone();
                    let elapsed = thread_start.elapsed();
                    found_flag_ref.store(true, Ordering::Relaxed);
                    let mut guard = result_ref.lock().unwrap();
                    if guard.is_none() {
                        *guard = Some(EnsembleHit {
                            factor,
                            algorithm: Algorithm::TrialDivision,
                            duration: elapsed,
                        });
                    }
                }
            });
        }

        // Spin-wait for a result or timeout
        while !found_flag.load(Ordering::Relaxed) {
            if start.elapsed() >= timeout {
                // Signal all threads to stop
                found_flag.store(true, Ordering::Relaxed);
                break;
            }
            std::thread::sleep(Duration::from_millis(1));
        }
    });

    // Extract the result
    let guard = result.lock().unwrap();
    match &*guard {
        Some(hit) => {
            let cofactor = n / &hit.factor;
            let mut factors = vec![hit.factor.clone(), cofactor];
            factors.sort();
            FactorResult {
                n: n.clone(),
                factors,
                algorithm: hit.algorithm,
                duration: hit.duration,
                complete: true,
            }
        }
        None => FactorResult {
            n: n.clone(),
            factors: vec![],
            algorithm: Algorithm::Ensemble,
            duration: start.elapsed(),
            complete: false,
        },
    }
}

/// Per-method outcome in a method comparison report.
pub struct MethodComparison {
    /// Each entry: (algorithm, optional factor found, time taken).
    pub results: Vec<(Algorithm, Option<BigUint>, Duration)>,
    /// The algorithm that succeeded fastest, if any.
    pub fastest_algorithm: Option<Algorithm>,
}

/// Run each factoring method sequentially with a per-method time limit,
/// and report which succeeded, which factor they found, and how long each took.
///
/// Methods tested:
///   1. Trial division (bound 1,000,000)
///   2. Pollard rho (Floyd)
///   3. Pollard rho (Brent)
///   4. Pollard p-1 (bound 10,000)
///   5. Pollard p-1 (bound 100,000)
///
/// Each method gets up to 5 seconds. The `fastest_algorithm` field is set
/// to the method that found a factor in the least time.
pub fn compare_methods(n: &BigUint) -> MethodComparison {
    let per_method_timeout = Duration::from_secs(5);
    let mut results: Vec<(Algorithm, Option<BigUint>, Duration)> = Vec::new();

    // 1. Trial division
    {
        let start = Instant::now();
        let factors = trial_division(n, 1_000_000);
        let elapsed = start.elapsed();
        let factor = if factors.len() >= 2 {
            Some(factors[0].clone())
        } else {
            None
        };
        results.push((Algorithm::TrialDivision, factor, elapsed));
    }

    // 2. Pollard rho (Floyd)
    {
        let start = Instant::now();
        // Use a limited attempt count based on timeout
        let factor = pollard_rho(n);
        let elapsed = start.elapsed();
        let factor = if elapsed <= per_method_timeout {
            factor
        } else {
            None
        };
        results.push((Algorithm::PollardRho, factor, elapsed));
    }

    // 3. Pollard rho (Brent)
    {
        let start = Instant::now();
        let factor = pollard_rho_brent(n);
        let elapsed = start.elapsed();
        let factor = if elapsed <= per_method_timeout {
            factor
        } else {
            None
        };
        results.push((Algorithm::PollardRhoBrent, factor, elapsed));
    }

    // 4. Pollard p-1 (bound 10,000)
    {
        let start = Instant::now();
        let factor = pollard_p_minus_1(n, 10_000);
        let elapsed = start.elapsed();
        let factor = if elapsed <= per_method_timeout {
            factor
        } else {
            None
        };
        results.push((Algorithm::PollardPMinus1, factor, elapsed));
    }

    // 5. Pollard p-1 (bound 100,000)
    {
        let start = Instant::now();
        let factor = pollard_p_minus_1(n, 100_000);
        let elapsed = start.elapsed();
        let factor = if elapsed <= per_method_timeout {
            factor
        } else {
            None
        };
        results.push((Algorithm::PollardPMinus1, factor, elapsed));
    }

    // Find fastest successful method
    let fastest_algorithm = results
        .iter()
        .filter(|(_, factor, _)| factor.is_some())
        .min_by_key(|(_, _, dur)| *dur)
        .map(|(algo, _, _)| *algo);

    MethodComparison {
        results,
        fastest_algorithm,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trial_division() {
        let n = BigUint::from(60u32);
        let factors = trial_division(&n, 100);
        assert_eq!(factors, vec![
            BigUint::from(2u32),
            BigUint::from(2u32),
            BigUint::from(3u32),
            BigUint::from(5u32),
        ]);
    }

    #[test]
    fn test_pollard_rho() {
        let n = BigUint::from(8051u32); // 83 * 97
        let factor = pollard_rho(&n);
        assert!(factor.is_some());
        let f = factor.unwrap();
        assert!(f == BigUint::from(83u32) || f == BigUint::from(97u32));
    }

    #[test]
    fn test_is_probably_prime() {
        assert!(is_probably_prime(&BigUint::from(7u32), 20));
        assert!(is_probably_prime(&BigUint::from(104729u32), 20));
        assert!(!is_probably_prime(&BigUint::from(100u32), 20));
        assert!(!is_probably_prime(&BigUint::from(1u32), 20));
    }

    #[test]
    fn test_rsa_target_generation() {
        let mut rng = rand::thread_rng();
        let target = generate_rsa_target(64, &mut rng);
        assert_eq!(&target.p * &target.q, target.n);
    }

    #[test]
    fn test_random_prime_bit_length() {
        let mut rng = rand::thread_rng();
        // Test that random_prime generates primes with exactly the requested bit length
        for bits in [16, 32, 50, 64, 100, 128] {
            for _ in 0..5 {
                let p = random_prime(bits, &mut rng);
                assert_eq!(
                    p.bits(),
                    bits as u64,
                    "random_prime({}) generated a {}-bit number: {}",
                    bits,
                    p.bits(),
                    p
                );
            }
        }
    }

    #[test]
    fn test_rsa_target_bit_size() {
        let mut rng = rand::thread_rng();
        // Test that generate_rsa_target produces semiprimes within ±1 bit of target
        for target_bits in [64, 100, 128, 200] {
            for _ in 0..3 {
                let target = generate_rsa_target(target_bits, &mut rng);
                let actual_bits = target.n.bits() as u32;
                assert!(
                    actual_bits >= target_bits - 1 && actual_bits <= target_bits,
                    "generate_rsa_target({}) produced {}-bit semiprime (expected {}-{})",
                    target_bits,
                    actual_bits,
                    target_bits - 1,
                    target_bits
                );
            }
        }
    }

    #[test]
    fn test_to_base() {
        let n = BigUint::from(255u32);
        assert_eq!(to_base(&n, 2), vec![1, 1, 1, 1, 1, 1, 1, 1]);
        assert_eq!(to_base(&n, 16), vec![15, 15]);
        assert_eq!(to_base(&n, 10), vec![2, 5, 5]);
    }

    #[test]
    fn test_to_rns() {
        let n = BigUint::from(100u32);
        let moduli = vec![3, 5, 7];
        let rns = to_rns(&n, &moduli);
        assert_eq!(rns, vec![1, 0, 2]); // 100 mod 3=1, mod 5=0, mod 7=2
    }

    #[test]
    fn test_entropy() {
        // All same digit: entropy = 0
        assert_eq!(entropy(&[1, 1, 1, 1], 2), 0.0);
        // Equal distribution in binary: entropy = 1.0
        let e = entropy(&[0, 1, 0, 1], 2);
        assert!((e - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_pollard_p_minus_1() {
        // 17000051 = 17 * 1000003
        // 17 - 1 = 16 = 2^4 is 2-smooth, so p-1 with a small bound should find 17
        let n = BigUint::from(17_000_051u64);
        let factor = pollard_p_minus_1(&n, 20);
        assert!(factor.is_some(), "Pollard p-1 should find a factor of 17000051");
        let f = factor.unwrap();
        assert!(
            f == BigUint::from(17u32) || f == BigUint::from(1_000_003u64),
            "Factor should be 17 or 1000003, got {}",
            f
        );
        // Verify it actually divides n
        assert!((&n % &f).is_zero(), "Found factor must divide n");
    }

    #[test]
    fn test_factor_auto_small() {
        // 15 = 3 * 5 — small number, should use trial division
        let n = BigUint::from(15u32);
        let result = factor_auto(&n);
        assert!(result.complete, "Should successfully factor 15");
        assert_eq!(result.algorithm, Algorithm::TrialDivision);
        // Trial division returns prime factorization: [3, 5]
        let product: BigUint = result.factors.iter().fold(BigUint::one(), |acc, f| acc * f);
        assert_eq!(product, n, "Product of factors should equal n");
        assert!(
            result.factors.contains(&BigUint::from(3u32)),
            "Factors of 15 should include 3"
        );
        assert!(
            result.factors.contains(&BigUint::from(5u32)),
            "Factors of 15 should include 5"
        );
    }

    #[test]
    fn test_factor_auto_medium() {
        // 48-bit semiprime: 100003 * 100019 = 10002200057
        // Both primes are smallish, so Pollard rho or p-1 should handle it
        let p = BigUint::from(100_003u64);
        let q = BigUint::from(100_019u64);
        let n = &p * &q;
        assert!(n.bits() > 32 && n.bits() <= 64, "Should be a medium-sized number (33-64 bits)");

        let result = factor_auto(&n);
        assert!(result.complete, "Should successfully factor the medium semiprime");
        assert_eq!(result.factors.len(), 2, "Should find exactly two factors");

        let product: BigUint = result.factors.iter().fold(BigUint::one(), |acc, f| acc * f);
        assert_eq!(product, n, "Product of factors should equal n");

        let mut sorted_factors = result.factors.clone();
        sorted_factors.sort();
        assert_eq!(sorted_factors[0], p, "Smaller factor should be 100003");
        assert_eq!(sorted_factors[1], q, "Larger factor should be 100019");
    }

    #[test]
    fn test_mod_inverse() {
        // 3^(-1) mod 7 = 5, because 3 * 5 = 15 ≡ 1 (mod 7)
        let a = BigUint::from(3u32);
        let m = BigUint::from(7u32);
        let inv = mod_inverse(&a, &m);
        assert!(inv.is_some(), "Inverse of 3 mod 7 should exist");
        assert_eq!(inv.unwrap(), BigUint::from(5u32), "3^(-1) mod 7 should be 5");

        // Verify: no inverse when gcd != 1
        let a2 = BigUint::from(6u32);
        let m2 = BigUint::from(9u32);
        assert!(
            mod_inverse(&a2, &m2).is_none(),
            "Inverse of 6 mod 9 should not exist (gcd=3)"
        );

        // Additional: verify a * a^(-1) ≡ 1 (mod m) for several cases
        let a3 = BigUint::from(17u32);
        let m3 = BigUint::from(43u32);
        let inv3 = mod_inverse(&a3, &m3).expect("17 and 43 are coprime");
        assert_eq!((&a3 * &inv3) % &m3, BigUint::one(), "17 * inv(17) should be 1 mod 43");
    }

    #[test]
    fn test_pollard_rho_brent() {
        // 8051 = 83 * 97
        let n = BigUint::from(8051u32);
        let factor = pollard_rho_brent(&n);
        assert!(factor.is_some(), "Brent's rho should find a factor of 8051");
        let f = factor.unwrap();
        assert!(
            f == BigUint::from(83u32) || f == BigUint::from(97u32),
            "Factor should be 83 or 97, got {}",
            f
        );
        // Verify it divides n
        assert!((&n % &f).is_zero(), "Found factor must divide n");
    }

    #[test]
    fn test_factor_ensemble() {
        // 32-bit semiprime: 1000003 * 1000033 = 1000036000099
        let p = BigUint::from(1_000_003u64);
        let q = BigUint::from(1_000_033u64);
        let n = &p * &q;

        let result = factor_ensemble(&n, 30_000); // 30 second timeout
        assert!(
            result.complete,
            "Ensemble should factor a 40-bit semiprime within timeout"
        );
        assert_eq!(result.factors.len(), 2, "Should find exactly two factors");

        let product: BigUint = result.factors.iter().fold(BigUint::one(), |acc, f| acc * f);
        assert_eq!(product, n, "Product of factors should equal n");

        // Verify the factors are the expected primes (sorted)
        let mut sorted = result.factors.clone();
        sorted.sort();
        assert_eq!(sorted[0], p, "Smaller factor should be 1000003");
        assert_eq!(sorted[1], q, "Larger factor should be 1000033");
    }

    #[test]
    fn test_compare_methods() {
        // Use a number that multiple methods can factor: 8051 = 83 * 97
        let n = BigUint::from(8051u32);
        let comparison = compare_methods(&n);

        // Should have results for all 5 method runs
        assert_eq!(
            comparison.results.len(),
            5,
            "Should have results for 5 method runs"
        );

        // At least one method should have succeeded
        let successful_count = comparison
            .results
            .iter()
            .filter(|(_, factor, _)| factor.is_some())
            .count();
        assert!(
            successful_count >= 1,
            "At least one method should find a factor of 8051"
        );

        // fastest_algorithm should be set
        assert!(
            comparison.fastest_algorithm.is_some(),
            "Should identify a fastest algorithm"
        );

        // Every successful result should return a valid factor
        for (algo, factor, _dur) in &comparison.results {
            if let Some(f) = factor {
                assert!(
                    (&n % f).is_zero(),
                    "Factor {} from {:?} should divide n",
                    f,
                    algo
                );
                assert!(
                    *f > BigUint::one() && *f < n,
                    "Factor from {:?} should be nontrivial",
                    algo
                );
            }
        }
    }
}
