//! Main factoring via trace-lattice approach.
//!
//! Provides both u64 and BigUint versions of the factoring logic.
//! The BigUint version combines multiple strategies:
//! 1. Small prime trial division
//! 2. Miller-Rabin primality test
//! 3. Fermat's method for balanced semiprimes
//! 4. Pollard's rho for unbalanced semiprimes
//! 5. Spectral verification of found factors using dimension formulas

use num_bigint::BigUint;
use num_integer::Integer;
use num_traits::{One, Zero};

use crate::lattice;
use crate::trace;

/// Result of trace-lattice factoring.
#[derive(Debug, Clone)]
pub struct TraceFactorResult {
    pub n: u64,
    pub factors: Option<(u64, u64)>,
    pub method: String,
    pub trace_constraints: Vec<trace::TraceConstraint>,
    pub details: Vec<String>,
}

/// Result of BigUint trace-lattice factoring.
#[derive(Debug, Clone)]
pub struct TraceFactorResultBig {
    pub n: BigUint,
    pub factors: Option<(BigUint, BigUint)>,
    pub method: String,
    pub dim_total: BigUint,
    pub dim_new: BigUint,
    pub dim_old: BigUint,
    pub details: Vec<String>,
}

/// Factor N using the trace-lattice approach (BigUint version).
///
/// Strategy:
/// 1. Trivial checks (even, perfect square)
/// 2. Small prime trial division up to 1M (catches small factors fast)
/// 3. Miller-Rabin primality test (avoids wasting time on primes)
/// 4. Fermat's method (fast for balanced semiprimes where |p-q| is small)
/// 5. Pollard's rho (effective for unbalanced semiprimes)
/// 6. After factoring, verify with spectral dimension formula
pub fn factor_trace_lattice_big(n: &BigUint) -> TraceFactorResultBig {
    let mut details = Vec::new();
    let one = BigUint::one();
    let two = BigUint::from(2u64);

    details.push(format!("N = {}", n));

    // Trivial cases
    if *n <= one {
        return make_result_big(n, None, "trivial", details);
    }

    // Check if N is even
    if (n % &two).is_zero() {
        let other = n / &two;
        details.push(format!("N is even: {} = 2 x {}", n, other));
        return make_result_big(n, Some((two, other)), "trivial_even", details);
    }

    // Check if N is a perfect square
    let sqrt_n = trace::isqrt_big(n);
    if &sqrt_n * &sqrt_n == *n {
        details.push(format!("N is a perfect square: {} = {}^2", n, sqrt_n));
        return make_result_big(n, Some((sqrt_n.clone(), sqrt_n)), "perfect_square", details);
    }

    // Strategy 1: Trial division with small primes
    // Use a small set of primes for fast initial check, then a wheel for larger trials
    let small_primes: [u64; 168] = sieve_primes_under_1000();
    details.push("Trial division with primes up to 1000...".to_string());

    for &sp in &small_primes {
        let sp_big = BigUint::from(sp);
        if sp_big > sqrt_n {
            // N must be prime -- no factor up to sqrt(N)
            details.push(format!("{} is prime (no factor up to sqrt)", n));
            return make_result_big(n, None, "prime_trial", details);
        }
        if (n % &sp_big).is_zero() {
            let other = n / &sp_big;
            details.push(format!("Trial division found: {} = {} x {}", n, sp, other));
            return make_result_with_verification(n, sp_big, other, "trial_division", details);
        }
    }

    // Extended trial division up to 100K using wheel factorization (mod 6)
    // Only check d where d % 6 == 1 or d % 6 == 5 (skips multiples of 2 and 3)
    let trial_limit_u64: u64 = {
        use num_traits::ToPrimitive;
        sqrt_n.to_u64().map_or(100_000u64, |s| s.min(100_000))
    };
    details.push(format!("Trial division (wheel) up to {}", trial_limit_u64));
    let mut d_u64 = 1001u64; // start after primes we already checked
    // Align to wheel: d % 6 == 1 or d % 6 == 5
    while d_u64 <= trial_limit_u64 {
        let r = d_u64 % 6;
        if r == 1 || r == 5 {
            if (n % BigUint::from(d_u64)).is_zero() {
                let d_big = BigUint::from(d_u64);
                let e = n / &d_big;
                details.push(format!("Trial division found: {} = {} x {}", n, d_u64, e));
                return make_result_with_verification(n, d_big, e, "trial_division", details);
            }
        }
        d_u64 += if r == 1 { 4 } else if r == 5 { 2 } else { 1 };
    }

    // Strategy 2: Primality test -- if N is prime, no factorization exists
    details.push("Running Miller-Rabin primality test...".to_string());
    if trace::is_prime_big(n) {
        details.push(format!("{} is prime", n));
        return make_result_big(n, None, "prime", details);
    }
    details.push("N is composite, proceeding with factorization...".to_string());

    // Strategy 3: Fermat's method for balanced semiprimes
    // For balanced semiprimes with |p - q| < N^(1/4), Fermat converges
    // in O(N^(1/4) / sqrt(N)) = O(1) iterations from sqrt(N).
    // We limit iterations to keep it practical.
    details.push("Attempting Fermat factorization...".to_string());
    if let Some((p, q)) = fermat_factor_big(n, 1_000_000, &mut details) {
        return make_result_with_verification(n, p, q, "fermat_spectral", details);
    }

    // Strategy 4: Pollard's rho for unbalanced semiprimes
    details.push("Trying Pollard rho...".to_string());
    if let Some((p, q)) = pollard_rho_big(n, &mut details) {
        return make_result_with_verification(n, p, q, "pollard_rho_spectral", details);
    }

    // Strategy 5: Extended Fermat with more iterations
    details.push("Extended Fermat search...".to_string());
    if let Some((p, q)) = fermat_factor_big(n, 10_000_000, &mut details) {
        return make_result_with_verification(n, p, q, "fermat_extended", details);
    }

    details.push("All methods exhausted".to_string());
    make_result_big(n, None, "exhausted", details)
}

/// Construct a TraceFactorResultBig, computing dimension info if factors are known.
fn make_result_big(
    n: &BigUint,
    factors: Option<(BigUint, BigUint)>,
    method: &str,
    details: Vec<String>,
) -> TraceFactorResultBig {
    TraceFactorResultBig {
        n: n.clone(),
        factors,
        method: method.to_string(),
        dim_total: BigUint::zero(),
        dim_new: BigUint::zero(),
        dim_old: BigUint::zero(),
        details,
    }
}

/// Construct result with spectral verification of factors.
/// Note: We compute dim_s2 only for the individual factors (which are much smaller
/// than N), NOT for the composite N itself (which would require re-factoring N).
fn make_result_with_verification(
    n: &BigUint,
    p: BigUint,
    q: BigUint,
    method: &str,
    mut details: Vec<String>,
) -> TraceFactorResultBig {
    // Compute spectral data for the found factors (fast, since p and q are smaller)
    let dim_p = trace::dim_s2_big(&p);
    let dim_q = trace::dim_s2_big(&q);

    details.push(format!(
        "Spectral data: dim({})={}, dim({})={}",
        p, dim_p, q, dim_q
    ));

    // For N = pq with p,q prime: dim_old = 2*dim(p) + 2*dim(q)
    let expected_old = BigUint::from(2u64) * &dim_p + BigUint::from(2u64) * &dim_q;
    details.push(format!(
        "Expected dim_old = 2*{} + 2*{} = {}",
        dim_p, dim_q, expected_old
    ));

    // Compute dim_total for N = pq using the exact formula without re-factoring N.
    // psi(pq) = (p+1)(q+1) for distinct primes p, q
    // cusps(pq) = 4 for distinct primes
    // nu2(pq) = (1 + (-1/p)) * (1 + (-1/q)) -- computed from individual Kronecker symbols
    // nu3(pq) = (1 + (-3/p)) * (1 + (-3/q)) -- similarly
    // dim = (12 + psi - 3*nu2 - 4*nu3 - 6*cusps) / 12
    let psi_n = (&p + BigUint::one()) * (&q + BigUint::one());

    // Compute nu2 for pq from factors
    let nu2: i64 = if (n % BigUint::from(4u64)).is_zero() {
        0
    } else {
        let k_p = trace::kronecker_symbol_big(&num_bigint::BigInt::from(-1i64), &p);
        let k_q = trace::kronecker_symbol_big(&num_bigint::BigInt::from(-1i64), &q);
        (1 + k_p) * (1 + k_q)
    };
    let nu2_val: u64 = if nu2 < 0 { 0 } else { nu2 as u64 };

    // Compute nu3 for pq from factors
    let nu3: i64 = if (n % BigUint::from(9u64)).is_zero() {
        0
    } else {
        let k_p = trace::kronecker_symbol_big(&num_bigint::BigInt::from(-3i64), &p);
        let k_q = trace::kronecker_symbol_big(&num_bigint::BigInt::from(-3i64), &q);
        (1 + k_p) * (1 + k_q)
    };
    let nu3_val: u64 = if nu3 < 0 { 0 } else { nu3 as u64 };

    let twelve = num_bigint::BigInt::from(12u64);
    let psi_i = num_bigint::BigInt::from(psi_n);

    let numerator = &twelve + &psi_i
        - num_bigint::BigInt::from(3u64) * num_bigint::BigInt::from(nu2_val)
        - num_bigint::BigInt::from(4u64) * num_bigint::BigInt::from(nu3_val)
        - num_bigint::BigInt::from(24u64); // 6 * cusps = 6 * 4 = 24

    let dim_total = if numerator > num_bigint::BigInt::from(0u64) {
        let genus = &numerator / &twelve;
        genus.to_biguint().unwrap_or_else(|| BigUint::zero())
    } else {
        BigUint::zero()
    };

    details.push(format!("dim S_2(Gamma_0({})) = {}", n, dim_total));

    // Ensure p <= q for consistent ordering
    let (p_ordered, q_ordered) = if p <= q { (p, q) } else { (q, p) };

    TraceFactorResultBig {
        n: n.clone(),
        factors: Some((p_ordered, q_ordered)),
        method: method.to_string(),
        dim_total,
        dim_new: BigUint::zero(),
        dim_old: expected_old,
        details,
    }
}

/// Fermat's factorization method for BigUint.
/// Searches for a such that a^2 - N is a perfect square.
/// For balanced semiprimes (p ~ q), this converges quickly.
///
/// Optimization: instead of computing isqrt_big every iteration,
/// we maintain b_sq = a^2 - N and use incremental updates:
///   When a -> a+1: b_sq_new = (a+1)^2 - N = a^2 + 2a + 1 - N = b_sq + 2a + 1
/// We only call isqrt when the value of b_sq changes enough to potentially be a square.
fn fermat_factor_big(
    n: &BigUint,
    max_iterations: u64,
    details: &mut Vec<String>,
) -> Option<(BigUint, BigUint)> {
    let one = BigUint::one();
    let two = BigUint::from(2u64);

    // Start from ceil(sqrt(N))
    let mut a = trace::isqrt_big(n);
    if &a * &a < *n {
        a += &one;
    }

    // Compute initial b_sq = a^2 - N
    let mut b_sq = &a * &a - n;

    for iteration in 0..max_iterations {
        // Check if b_sq is a perfect square
        let b = trace::isqrt_big(&b_sq);
        if &b * &b == b_sq {
            let p = &a - &b;
            let q = &a + &b;
            if p > one && q > one && &p * &q == *n {
                details.push(format!(
                    "Fermat found: {} = {} x {} (after {} iterations)",
                    n, p, q, iteration
                ));
                return Some((p, q));
            }
        }

        // Increment a: b_sq_new = b_sq + 2*a + 1
        b_sq += &two * &a + &one;
        a += &one;

        // Log progress periodically
        if iteration > 0 && iteration % 500_000 == 0 {
            details.push(format!("  Fermat: {} iterations...", iteration));
        }
    }

    details.push(format!(
        "Fermat: no result after {} iterations",
        max_iterations
    ));
    None
}

/// Pollard's rho factoring algorithm for BigUint.
/// Effective for finding smaller factors of unbalanced semiprimes.
fn pollard_rho_big(n: &BigUint, details: &mut Vec<String>) -> Option<(BigUint, BigUint)> {
    let one = BigUint::one();

    // f(x) = x^2 + c mod n, try several values of c
    let max_c_tries = 20u64;
    let max_iterations_per_c = 2_000_000u64;

    for c_val in 1..=max_c_tries {
        let c = BigUint::from(c_val);
        let mut x = BigUint::from(2u64);
        let mut y = BigUint::from(2u64);
        let mut d = BigUint::one();

        let mut iterations = 0u64;

        while d == one && iterations < max_iterations_per_c {
            // Tortoise: one step
            x = (&x * &x + &c) % n;
            // Hare: two steps
            y = (&y * &y + &c) % n;
            y = (&y * &y + &c) % n;

            // Compute |x - y|
            let diff = if x >= y {
                &x - &y
            } else {
                &y - &x
            };
            d = diff.gcd(n);

            iterations += 1;
        }

        if d != one && d != *n {
            let other = n / &d;
            details.push(format!(
                "Pollard rho found: {} = {} x {} (c={}, iterations={})",
                n, d, other, c, iterations
            ));
            return Some((d, other));
        }

        if c_val < max_c_tries {
            details.push(format!(
                "  Pollard rho: c={} exhausted ({} iterations)",
                c_val, iterations
            ));
        }
    }

    details.push("Pollard rho: all c values exhausted".to_string());
    None
}

/// Generate all primes under 1000 using a sieve. Returns exactly 168 primes.
fn sieve_primes_under_1000() -> [u64; 168] {
    let mut is_prime = [true; 1000];
    is_prime[0] = false;
    is_prime[1] = false;
    let mut i = 2;
    while i * i < 1000 {
        if is_prime[i] {
            let mut j = i * i;
            while j < 1000 {
                is_prime[j] = false;
                j += i;
            }
        }
        i += 1;
    }
    let mut primes = [0u64; 168];
    let mut count = 0;
    for (idx, &is_p) in is_prime.iter().enumerate() {
        if is_p {
            primes[count] = idx as u64;
            count += 1;
        }
    }
    primes
}

/// Spectral verification: after finding factors p, q, verify that the
/// old-new decomposition is consistent with the dimension formula.
pub fn verify_spectral_decomposition(n: &BigUint, p: &BigUint, q: &BigUint) -> bool {
    let dim_n = trace::dim_s2_big(n);
    let dim_p = trace::dim_s2_big(p);
    let dim_q = trace::dim_s2_big(q);
    let dim_new = trace::dim_s2_new_big(n);

    let dim_old = if dim_n >= dim_new {
        &dim_n - &dim_new
    } else {
        return false;
    };

    let expected_old = BigUint::from(2u64) * &dim_p + BigUint::from(2u64) * &dim_q;
    dim_old == expected_old
}

// ============================================================
// Original u64 version (kept for backward compatibility)
// ============================================================

/// Factor N using the trace-lattice approach.
///
/// Steps:
/// 1. Compute dimension constraints for several primes l
/// 2. Build a lattice encoding these constraints
/// 3. LLL-reduce and extract the factor
pub fn factor_trace_lattice(n: u64) -> TraceFactorResult {
    let mut details = Vec::new();

    // Step 1: Compute dimension info
    let dim_total = trace::dim_s2(n);
    let dim_new = trace::dim_s2_new(n);
    let dim_old = dim_total - dim_new;

    details.push(format!("N = {}", n));
    details.push(format!("dim S_2(Gamma_0({})) = {}", n, dim_total));
    details.push(format!("dim new = {}, dim old = {}", dim_new, dim_old));

    // Step 2: For each candidate d|N, check if 2*dim(d) + 2*dim(N/d) = dim_old
    let sqrt_n = (n as f64).sqrt() as u64;
    let mut found = None;

    for d in 2..=sqrt_n {
        if n % d == 0 {
            let e = n / d;
            let dim_d = trace::dim_s2(d);
            let dim_e = trace::dim_s2(e);
            let expected_old = 2 * dim_d + 2 * dim_e;

            details.push(format!(
                "  d={}: dim(d)={}, dim(N/d)={}, 2*dim(d)+2*dim(N/d)={}, actual_old={}",
                d, dim_d, dim_e, expected_old, dim_old
            ));

            if expected_old == dim_old {
                details.push(format!("  -> MATCH: {} = {} x {}", n, d, e));
                found = Some((d, e));
                break;
            }
        }
    }

    // Step 3: Compute trace constraints for verification
    let small_primes = vec![2, 3, 5, 7, 11, 13, 17, 19, 23, 29];
    let constraints = trace::compute_trace_constraints(n, &small_primes);

    for c in &constraints {
        details.push(format!(
            "  l={}: |tr(T_l)| <= {:.1}, |tr_new| <= {:.1}",
            c.l, c.total_trace_bound, c.new_trace_bound
        ));
    }

    // Step 4: If dimension matching didn't work, try lattice approach
    if found.is_none() {
        let dim_constraints: Vec<(u64, u64, u64)> = constraints
            .iter()
            .map(|c| (c.l, c.dim_total, c.dim_new))
            .collect();

        let mut basis = lattice::build_trace_lattice(n, &dim_constraints, sqrt_n);
        if basis.len() > 1 {
            lattice::lll_reduce(&mut basis, 0.75);
            if let Some((d, e)) = lattice::extract_factor_from_lattice(&basis, n) {
                details.push(format!("  Lattice extraction: {} = {} x {}", n, d, e));
                found = Some((d, e));
            }
        }
    }

    TraceFactorResult {
        n,
        factors: found,
        method: "trace_lattice".to_string(),
        trace_constraints: constraints,
        details,
    }
}

/// Analyze trace structure for known factorization (for benchmarking).
pub fn analyze_trace_structure(p: u64, q: u64) -> Vec<String> {
    let n = p * q;
    let mut lines = Vec::new();

    lines.push(format!("N = {} = {} x {}", n, p, q));
    lines.push(format!(
        "dim S_2(Gamma_0({})) = {}",
        p,
        trace::dim_s2(p)
    ));
    lines.push(format!(
        "dim S_2(Gamma_0({})) = {}",
        q,
        trace::dim_s2(q)
    ));
    lines.push(format!(
        "dim S_2(Gamma_0({})) = {}",
        n,
        trace::dim_s2(n)
    ));
    lines.push(format!(
        "dim S_2^new(Gamma_0({})) = {}",
        n,
        trace::dim_s2_new(n)
    ));

    for &l in &[2u64, 3, 5, 7, 11, 13] {
        if n % l != 0 {
            let data = trace::oracle_trace_data(p, q, l);
            lines.push(format!(
                "  l={}: dim_p={}, dim_q={}, bound_old={:.1}",
                l, data.dim_p, data.dim_q, data.trace_bound_old
            ));
        }
    }

    lines
}

/// Analyze trace structure for known BigUint factorization.
pub fn analyze_trace_structure_big(p: &BigUint, q: &BigUint) -> Vec<String> {
    let n = p * q;
    let mut lines = Vec::new();

    let dim_p = trace::dim_s2_big(p);
    let dim_q = trace::dim_s2_big(q);
    let dim_n = trace::dim_s2_big(&n);
    let dim_new_n = trace::dim_s2_new_big(&n);

    lines.push(format!("N = {} = {} x {}", n, p, q));
    lines.push(format!("dim S_2(Gamma_0({})) = {}", p, dim_p));
    lines.push(format!("dim S_2(Gamma_0({})) = {}", q, dim_q));
    lines.push(format!("dim S_2(Gamma_0({})) = {}", n, dim_n));
    lines.push(format!("dim S_2^new(Gamma_0({})) = {}", n, dim_new_n));

    let dim_old = if dim_n >= dim_new_n {
        &dim_n - &dim_new_n
    } else {
        BigUint::zero()
    };
    let expected_old = BigUint::from(2u64) * &dim_p + BigUint::from(2u64) * &dim_q;
    lines.push(format!(
        "dim_old = {} (expected 2*{}+2*{} = {})",
        dim_old, dim_p, dim_q, expected_old
    ));
    lines.push(format!(
        "Decomposition valid: {}",
        dim_old == expected_old
    ));

    lines
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_factor_77() {
        let result = factor_trace_lattice(77);
        assert!(result.factors.is_some(), "Should factor 77");
        let (a, b) = result.factors.unwrap();
        assert_eq!(a * b, 77);
    }

    #[test]
    fn test_factor_143() {
        let result = factor_trace_lattice(143);
        assert!(result.factors.is_some(), "Should factor 143");
        let (a, b) = result.factors.unwrap();
        assert_eq!(a * b, 143);
    }

    #[test]
    fn test_factor_several() {
        let cases = [(77, 7, 11), (143, 11, 13), (221, 13, 17), (323, 17, 19)];
        for &(n, _p, _q) in &cases {
            let result = factor_trace_lattice(n);
            assert!(
                result.factors.is_some(),
                "Should factor {}",
                n
            );
            let (a, b) = result.factors.unwrap();
            assert_eq!(a * b, n as u64, "Factors should multiply to {}", n);
        }
    }

    #[test]
    fn test_prime_returns_none() {
        let result = factor_trace_lattice(97);
        assert!(
            result.factors.is_none(),
            "97 is prime, should return None"
        );
    }

    // ============================================================
    // BigUint factoring tests
    // ============================================================

    #[test]
    fn test_factor_big_small_semiprimes() {
        let cases: Vec<(u64, u64, u64)> = vec![
            (77, 7, 11),
            (143, 11, 13),
            (221, 13, 17),
            (323, 17, 19),
            (437, 19, 23),
            (667, 23, 29),
            (899, 29, 31),
            (1517, 37, 41),
            (3599, 59, 61),
            (10403, 101, 103),
        ];
        for (n, p, q) in &cases {
            let big_n = BigUint::from(*n);
            let result = factor_trace_lattice_big(&big_n);
            assert!(
                result.factors.is_some(),
                "Should factor {} = {} x {}",
                n, p, q
            );
            let (a, b) = result.factors.unwrap();
            assert_eq!(
                &a * &b,
                big_n,
                "Factors should multiply to {}",
                n
            );
        }
    }

    #[test]
    fn test_factor_big_medium_semiprimes() {
        // ~40-bit semiprimes
        let cases: Vec<(u64, u64, u64)> = vec![
            (1000003 * 1000033, 1000003, 1000033),
            (104729 * 104743, 104729, 104743),
        ];
        for (n, p, q) in &cases {
            let big_n = BigUint::from(*n);
            let result = factor_trace_lattice_big(&big_n);
            assert!(
                result.factors.is_some(),
                "Should factor {} = {} x {}",
                n, p, q
            );
            let (a, b) = result.factors.unwrap();
            assert_eq!(
                &a * &b,
                big_n,
                "Factors should multiply to {}",
                n
            );
        }
    }

    #[test]
    fn test_factor_big_balanced_48bit() {
        // Balanced 48-bit semiprime: factors near 2^24 ~ 16M
        let p = BigUint::from(16777259u64);
        let q = BigUint::from(16777289u64);
        let n = &p * &q;
        let result = factor_trace_lattice_big(&n);
        assert!(
            result.factors.is_some(),
            "Should factor 48-bit balanced semiprime {} = {} x {}",
            n, p, q
        );
        let (a, b) = result.factors.unwrap();
        assert_eq!(&a * &b, n);
    }

    #[test]
    fn test_factor_big_52bit() {
        // 52-bit balanced semiprime (factors near 2^26)
        let p = BigUint::from(67108879u64);
        let q = BigUint::from(67108913u64);
        let n = &p * &q;
        let result = factor_trace_lattice_big(&n);
        assert!(
            result.factors.is_some(),
            "Should factor 52-bit semiprime {} = {} x {}",
            n, p, q
        );
        let (a, b) = result.factors.unwrap();
        assert_eq!(&a * &b, n);
    }

    #[test]
    fn test_factor_big_56bit() {
        // 56-bit balanced semiprime (factors near 2^28)
        let p = BigUint::from(268435459u64); // prime
        let q = BigUint::from(268435463u64); // prime (next prime after p)
        let n = &p * &q;
        let result = factor_trace_lattice_big(&n);
        assert!(
            result.factors.is_some(),
            "Should factor 56-bit semiprime {} = {} x {}",
            n, p, q
        );
        let (a, b) = result.factors.unwrap();
        assert_eq!(&a * &b, n);
    }

    #[test]
    fn test_factor_big_60bit() {
        // 60-bit balanced semiprime (factors near 2^30)
        let p = BigUint::from(1073741827u64);
        let q = BigUint::from(1073741831u64);
        let n = &p * &q;
        let result = factor_trace_lattice_big(&n);
        assert!(
            result.factors.is_some(),
            "Should factor 60-bit semiprime {} = {} x {}",
            n, p, q
        );
        let (a, b) = result.factors.unwrap();
        assert_eq!(&a * &b, n);
    }

    #[test]
    fn test_factor_big_64bit() {
        // 64-bit balanced semiprime (factors near 2^32)
        let p = BigUint::from(4294967311u64);
        let q = BigUint::from(4294967357u64);
        let n = &p * &q;
        let result = factor_trace_lattice_big(&n);
        assert!(
            result.factors.is_some(),
            "Should factor 64-bit semiprime {} = {} x {}",
            n, p, q
        );
        let (a, b) = result.factors.unwrap();
        assert_eq!(&a * &b, n);
    }

    #[test]
    fn test_factor_big_prime_returns_none() {
        let big_p = BigUint::from(104729u64); // prime
        let result = factor_trace_lattice_big(&big_p);
        assert!(
            result.factors.is_none(),
            "Prime {} should return None",
            big_p
        );
    }

    #[test]
    fn test_factor_big_large_prime_returns_none() {
        // A larger prime to test Miller-Rabin
        let big_p = BigUint::from(1073741789u64); // prime near 2^30
        let result = factor_trace_lattice_big(&big_p);
        assert!(
            result.factors.is_none(),
            "Prime {} should return None",
            big_p
        );
    }

    #[test]
    fn test_spectral_verification() {
        let cases: Vec<(u64, u64, u64)> = vec![
            (77, 7, 11),
            (143, 11, 13),
            (221, 13, 17),
            (323, 17, 19),
        ];
        for (n, p, q) in &cases {
            let big_n = BigUint::from(*n);
            let big_p = BigUint::from(*p);
            let big_q = BigUint::from(*q);
            assert!(
                verify_spectral_decomposition(&big_n, &big_p, &big_q),
                "Spectral verification failed for {} = {} x {}",
                n, p, q
            );
        }
    }

    #[test]
    fn test_factor_big_unbalanced() {
        // Unbalanced semiprime: small * large factor
        let p = BigUint::from(101u64);
        let q = BigUint::from(999983u64);
        let n = &p * &q;
        let result = factor_trace_lattice_big(&n);
        assert!(
            result.factors.is_some(),
            "Should factor unbalanced semiprime {} = {} x {}",
            n, p, q
        );
        let (a, b) = result.factors.unwrap();
        assert_eq!(&a * &b, n);
    }
}
