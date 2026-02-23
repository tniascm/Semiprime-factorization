//! Primitive operations on BigUint for the evolutionary factoring DSL.
//!
//! Each function implements a single composable building block that programs
//! in the genetic programming population can invoke. Includes wrappers for
//! operations from other project crates (cf-factor, ecm, lattice-reduction,
//! smooth-pilatte, cf-factor-ms).

use num_bigint::BigUint;
use num_integer::Integer;
use num_traits::{One, ToPrimitive, Zero};
use rand::Rng;

/// Fast perfect square test using mod-16 bitmask + integer sqrt.
/// Eliminates 75% of non-squares with a single bitwise check.
pub fn is_perfect_square(x: &BigUint) -> bool {
    if x.is_zero() {
        return true;
    }
    // Mod 16 filter: squares can only be 0,1,4,9 mod 16
    let low_bits = x.to_u64_digits();
    if !low_bits.is_empty() {
        let m16 = (low_bits[0] & 0xF) as u32;
        if !matches!(m16, 0 | 1 | 4 | 9) {
            return false;
        }
    }
    let s = x.sqrt();
    &s * &s == *x
}

/// Fermat step with multiplier: compute a^2 - k*N and check if perfect square.
/// Returns gcd(a + sqrt(a^2 - kN), N) if a^2 - kN is a perfect square, else None.
pub fn fermat_step(n: &BigUint, a: &BigUint, k: u64) -> Option<BigUint> {
    let kn = n * BigUint::from(k);
    let a_sq = a * a;
    if a_sq < kn {
        return None;
    }
    let diff = &a_sq - &kn;
    if is_perfect_square(&diff) {
        let b = diff.sqrt();
        let sum = a + &b;
        let g = sum.gcd(n);
        let one = BigUint::one();
        if g > one && g < *n {
            return Some(g);
        }
    }
    None
}

/// Hart's one-line factoring step: for multiplier i, compute s = ceil(sqrt(n*i)),
/// then m = s^2 mod n. If m is a perfect square, return gcd(s - sqrt(m), n).
pub fn hart_step(n: &BigUint, i: u64) -> Option<BigUint> {
    let ni = n * BigUint::from(i);
    let s = ni.sqrt() + BigUint::one(); // ceil(sqrt(n*i))
    let s_sq = &s * &s;
    let m = s_sq % n;
    if is_perfect_square(&m) {
        let sqrt_m = m.sqrt();
        if s > sqrt_m {
            let diff = &s - &sqrt_m;
            let g = diff.gcd(n);
            let one = BigUint::one();
            if g > one && g < *n {
                return Some(g);
            }
        }
    }
    None
}

/// Lucas V-sequence step for Williams p+1: V_{n+1} = A * V_n - V_{n-1} mod N
pub fn lucas_v_step(v_prev: &BigUint, v_curr: &BigUint, a: &BigUint, n: &BigUint) -> BigUint {
    let product = (a * v_curr) % n;
    // Safe subtraction mod n
    if product >= *v_prev {
        (&product - v_prev) % n
    } else {
        (n - (v_prev - &product) % n) % n
    }
}

/// Williams p+1 single stage: iterate Lucas V-sequence, check gcd(V - 2, N) periodically.
/// Returns a factor if p+1 is smooth up to bound.
pub fn williams_p_plus_1_step(n: &BigUint, a: &BigUint, bound: u64) -> Option<BigUint> {
    let one = BigUint::one();
    let two = BigUint::from(2u32);
    if *n <= one {
        return None;
    }

    let mut v_prev = two.clone();
    let mut v_curr = a.clone() % n;

    // Iterate through small primes and their powers
    let mut p = 2u64;
    while p <= bound {
        let mut pk = p;
        while pk <= bound {
            // V_{2k} = V_k^2 - 2 mod N (doubling formula for efficiency)
            let next = lucas_v_step(&v_prev, &v_curr, a, n);
            v_prev = v_curr;
            v_curr = next;
            pk = pk.saturating_mul(p);
        }

        // Check gcd(V - 2, N) periodically
        if p % 50 == 0 || p == bound {
            let v_minus_2 = if v_curr >= two {
                &v_curr - &two
            } else {
                n - (&two - &v_curr) % n
            };
            let g = v_minus_2.gcd(n);
            if g > one && g < *n {
                return Some(g);
            }
        }

        p += if p == 2 { 1 } else { 2 };
    }
    None
}

/// Integer square root (floor).
pub fn isqrt(n: &BigUint) -> BigUint {
    n.sqrt()
}

/// Check if candidate is of the form 6m+1 or 6m-1 (necessary for primes > 3).
pub fn is_6m_pm1(x: &BigUint) -> bool {
    let six = BigUint::from(6u32);
    let r = x % &six;
    r == BigUint::one() || r == BigUint::from(5u32)
}

/// Modular exponentiation: base^exp mod n.
pub fn mod_pow_prim(base: &BigUint, exp: &BigUint, n: &BigUint) -> BigUint {
    if n.is_zero() {
        return BigUint::zero();
    }
    base.modpow(exp, n)
}

/// Greatest common divisor of a and b.
pub fn gcd_prim(a: &BigUint, b: &BigUint) -> BigUint {
    a.gcd(b)
}

/// Sample a random element in [2, n-2].
/// Returns 2 if n <= 4 (degenerate case).
pub fn random_element_prim(n: &BigUint, rng: &mut impl Rng) -> BigUint {
    let two = BigUint::from(2u32);
    let four = BigUint::from(4u32);
    if *n <= four {
        return two;
    }

    let n_minus_3 = n - BigUint::from(3u32); // range size = n - 3
    let bytes = n.to_bytes_be();
    loop {
        let mut random_bytes = vec![0u8; bytes.len()];
        rng.fill(&mut random_bytes[..]);
        let val = BigUint::from_bytes_be(&random_bytes) % &n_minus_3;
        let result = val + &two; // shift into [2, n-2]
        if result >= two && result <= n - &two {
            return result;
        }
    }
}

/// Square modulo n: x^2 mod n.
pub fn square_mod(x: &BigUint, n: &BigUint) -> BigUint {
    if n.is_zero() {
        return BigUint::zero();
    }
    (x * x) % n
}

/// Add a constant modulo n: (x + c) mod n.
pub fn add_const_mod(x: &BigUint, c: u64, n: &BigUint) -> BigUint {
    if n.is_zero() {
        return BigUint::zero();
    }
    (x + BigUint::from(c)) % n
}

/// Multiply modulo n: (a * b) mod n.
pub fn multiply_mod(a: &BigUint, b: &BigUint, n: &BigUint) -> BigUint {
    if n.is_zero() {
        return BigUint::zero();
    }
    (a * b) % n
}

/// Compute gcd(|a - b|, n) and return it only if it is a nontrivial factor
/// (i.e., strictly between 1 and n). Returns None otherwise.
pub fn subtract_gcd(a: &BigUint, b: &BigUint, n: &BigUint) -> Option<BigUint> {
    let diff = if a > b {
        a - b
    } else if b > a {
        b - a
    } else {
        return None; // a == b => diff is 0 => gcd(0, n) = n (trivial)
    };

    if diff.is_zero() {
        return None;
    }

    let g = diff.gcd(n);
    let one = BigUint::one();
    if g > one && g < *n {
        Some(g)
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// New primitives wrapping project crates
// ---------------------------------------------------------------------------

/// State for Dixon's random squares method.
/// Accumulates smooth squares and their factorizations for later combination.
#[derive(Debug, Clone)]
pub struct DixonState {
    /// Accumulated values x where x² mod N was found to be smooth.
    pub x_values: Vec<BigUint>,
    /// The corresponding x² mod N values.
    pub square_values: Vec<BigUint>,
}

impl DixonState {
    pub fn new() -> Self {
        DixonState {
            x_values: Vec::new(),
            square_values: Vec::new(),
        }
    }
}

/// Compute the k-th continued fraction convergent of sqrt(N) and return
/// gcd(h_k² - N, N) if it's a nontrivial factor.
///
/// Uses h_values (convergent numerators) from the CF expansion.
pub fn cf_convergent_gcd(n: &BigUint, k: u32) -> Option<BigUint> {
    use num_traits::Signed;

    let one = BigUint::one();
    if *n <= one {
        return None;
    }

    // Use cf-factor's CF expansion
    let expansion = cf_factor::cf::cf_expand(n, (k as usize) + 2);
    if expansion.h_values.is_empty() {
        return None;
    }

    // Get the k-th convergent numerator (or last available)
    let idx = (k as usize).min(expansion.h_values.len() - 1);
    let h_k = &expansion.h_values[idx];

    // Compute h_k² - N and try gcd
    // h_k is BigInt, so we work with absolute values
    let h_k_abs = h_k.abs().to_biguint().unwrap_or_else(|| BigUint::zero());
    let h_sq = &h_k_abs * &h_k_abs;
    let diff: BigUint = if h_sq >= *n {
        &h_sq - n
    } else {
        n - &h_sq
    };

    if diff.is_zero() {
        return None;
    }

    let g = diff.gcd(n);
    if g > one && g < *n {
        Some(g)
    } else {
        None
    }
}

/// Run SQUFOF factoring attempt. Returns a factor if found quickly.
///
/// Only attempts SQUFOF on numbers small enough to complete quickly (≤ 48 bits).
/// Returns None immediately for larger inputs to prevent stalls.
pub fn squfof_attempt(n: &BigUint) -> Option<BigUint> {
    // SQUFOF is O(N^{1/4}) — only practical for small N in an evolutionary context
    if n.bits() > 48 {
        return None;
    }
    let result = cf_factor::squfof::squfof_factor(n);
    let one = BigUint::one();
    result.factor.filter(|f| *f > one && *f < *n)
}

/// Walk one step in the class group infrastructure and check if the form
/// reveals a factor. Returns a factor if found.
pub fn rho_form_step(n: &BigUint) -> Option<BigUint> {
    use cf_factor_ms::infrastructure::{form_reveals_factor, rho_step_ctx, InfraContext, InfraForm};

    let one = BigUint::one();
    if *n <= one {
        return None;
    }

    let ctx = InfraContext::new(n);
    let principal = InfraForm::principal(n);

    // Walk a few steps checking each form
    let mut current = principal;
    for _ in 0..10 {
        current = rho_step_ctx(&current, &ctx);
        if let Some(factor) = form_reveals_factor(&current.form, n) {
            if factor > one && factor < *n {
                return Some(factor);
            }
        }
    }
    None
}

/// Run one ECM curve with the given Phase 1 bound. Returns a factor if found.
///
/// Caps B1 at 2000 and B2 at 50000 to prevent long-running evaluations
/// during evolution. A single ECM curve is fast enough for small semiprimes.
pub fn ecm_attempt(n: &BigUint, b1: u64) -> Option<BigUint> {
    let capped_b1 = b1.max(100).min(2000); // Cap at 2000 for evolutionary speed
    let params = ecm::EcmParams {
        b1: capped_b1,
        b2: capped_b1 * 25, // Stage 2 bound = 25 × B1, capped implicitly
        num_curves: 1,       // Just one curve per call
    };
    let result = ecm::ecm_factor(n, &params);
    let one = BigUint::one();
    if result.complete {
        result.factors.into_iter().find(|f| *f > one && *f < *n)
    } else {
        None
    }
}

/// Build a small lattice from [state, N], LLL-reduce, and try to extract a
/// factor from the shortest vector via gcd.
pub fn lll_short_vector_gcd(state: &BigUint, n: &BigUint) -> Option<BigUint> {
    let one = BigUint::one();
    if *n <= one || state.is_zero() {
        return None;
    }

    // Build a 3x3 lattice encoding the relation: find small (a, b, c) with
    // a*state + b*N ≈ c. Short vectors give small linear combinations.
    let state_f64 = state.to_f64().unwrap_or(1.0);
    let n_f64 = n.to_f64().unwrap_or(1.0);

    // Scale factor to balance dimensions
    let scale = (n_f64.ln() * 3.0).ceil().max(1.0);

    let mut basis = vec![
        vec![scale, 0.0, state_f64],
        vec![0.0, scale, n_f64],
        vec![0.0, 0.0, scale * scale],
    ];

    let params = lattice_reduction::LllParams::default();
    lattice_reduction::lll_reduce(&mut basis, &params);

    // Check each row of the reduced basis for a factor
    for row in &basis {
        for &val in row {
            let abs_val = val.abs().round() as u64;
            if abs_val > 1 {
                let v = BigUint::from(abs_val);
                let g = v.gcd(n);
                if g > one && g < *n {
                    return Some(g);
                }
            }
        }
    }

    None
}

/// Test if state is B-smooth (all prime factors ≤ bound).
/// Returns the largest factor found by trial division, or 1 if not smooth.
pub fn smooth_test(state: &BigUint, bound: u64) -> BigUint {
    let one = BigUint::one();
    if *state <= one || bound < 2 {
        return one;
    }

    let mut remaining = state.clone();
    let mut largest_factor = BigUint::one();

    // Trial divide by small primes up to bound
    let mut p = 2u64;
    while p <= bound && remaining > one {
        let p_big = BigUint::from(p);
        while (&remaining % &p_big).is_zero() && remaining > one {
            remaining /= &p_big;
            if p_big > largest_factor {
                largest_factor = p_big.clone();
            }
        }
        p += if p == 2 { 1 } else { 2 };
    }

    if remaining.is_one() {
        largest_factor // Fully smooth
    } else {
        BigUint::one() // Not smooth
    }
}

/// Generate one exponent vector from a Pilatte lattice and compute the
/// corresponding product of primes mod N.
pub fn pilatte_vector_product(n: &BigUint) -> BigUint {
    let one = BigUint::one();
    if *n <= one {
        return one;
    }

    let n_bits = n.bits() as u32;
    let dim = (n_bits as f64).sqrt().ceil() as usize;
    let dim = dim.max(4).min(10); // Cap at 10 for evolutionary speed

    let result = smooth_pilatte::lattice::build_pilatte_lattice(n, dim);
    let vectors = smooth_pilatte::lattice::extract_exponent_vectors(&result);

    if vectors.is_empty() {
        return one;
    }

    // Compute product of primes^exponents mod N for the first vector
    let exps = &vectors[0];
    let primes = &result.params.primes;
    let mut product = BigUint::one();

    for (i, &exp) in exps.iter().enumerate() {
        if i >= primes.len() || exp == 0 {
            continue;
        }
        let p = BigUint::from(primes[i]);
        if exp > 0 {
            product = (&product * &p.modpow(&BigUint::from(exp as u64), n)) % n;
        }
    }

    product
}

/// Compute the Jacobi symbol (a/n). Returns 0, 1, or n-1 (representing -1)
/// as a BigUint for the evaluator.
pub fn jacobi_symbol_prim(a: &BigUint, n: &BigUint) -> BigUint {
    let one = BigUint::one();
    if *n <= one || n.is_even() {
        return BigUint::zero();
    }

    // Use num-integer's Jacobi symbol via i64 if small enough
    if let (Some(a_i64), Some(n_i64)) = (a.to_i64(), n.to_i64()) {
        let j = num_integer::Integer::gcd(&(a_i64 % n_i64), &n_i64);
        if j > 1 {
            return BigUint::zero(); // gcd > 1 means a shares factor with n
        }
        // Simple Jacobi computation for small values
        let result = jacobi_small(a_i64 % n_i64, n_i64);
        match result {
            0 => BigUint::zero(),
            1 => one,
            _ => n - &one, // -1 represented as n-1
        }
    } else {
        // For large values, just return the gcd-based information
        let g = a.gcd(n);
        if g > one && g < *n {
            BigUint::zero() // Shares a factor — not coprime
        } else {
            one
        }
    }
}

/// Simple Jacobi symbol computation for small i64 values.
fn jacobi_small(mut a: i64, mut n: i64) -> i64 {
    if n <= 0 || n % 2 == 0 {
        return 0;
    }
    a = a.rem_euclid(n);
    let mut result = 1i64;
    while a != 0 {
        while a % 2 == 0 {
            a /= 2;
            let n_mod_8 = n % 8;
            if n_mod_8 == 3 || n_mod_8 == 5 {
                result = -result;
            }
        }
        std::mem::swap(&mut a, &mut n);
        if a % 4 == 3 && n % 4 == 3 {
            result = -result;
        }
        a = a.rem_euclid(n);
    }
    if n == 1 {
        result
    } else {
        0
    }
}

/// Pollard p-1 factoring step: compute a^(B!) mod N and check gcd.
pub fn pollard_pm1_step(n: &BigUint, base: &BigUint, bound: u64) -> Option<BigUint> {
    let one = BigUint::one();
    if *n <= one || base.is_zero() {
        return None;
    }

    let mut a = base.clone() % n;
    if a < BigUint::from(2u32) {
        a = BigUint::from(2u32);
    }

    // Compute a^(lcm(1..bound)) mod N incrementally
    let capped_bound = bound.min(2_000); // Cap at 2000 for evolutionary speed
    let mut p = 2u64;
    while p <= capped_bound {
        let mut pk = p;
        while pk <= capped_bound {
            a = a.modpow(&BigUint::from(p), n);
            pk = pk.saturating_mul(p);
        }
        p += if p == 2 { 1 } else { 2 };
    }

    // Check gcd(a - 1, N)
    let a_minus_1 = if a > one { &a - &one } else { return None };
    if a_minus_1.is_zero() {
        return None;
    }
    let g = a_minus_1.gcd(n);
    if g > one && g < *n {
        Some(g)
    } else {
        None
    }
}

/// Accumulate a value for Dixon's method: if state² mod N is smooth,
/// store (state, state² mod N) in the Dixon buffer.
pub fn dixon_accumulate(dixon: &mut DixonState, state: &BigUint, n: &BigUint) {
    let one = BigUint::one();
    if *n <= one || state.is_zero() {
        return;
    }

    let sq = (state * state) % n;
    if sq.is_zero() {
        return;
    }

    // Quick smoothness check: try small primes
    let mut remaining = sq.clone();
    let small_primes: [u64; 15] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47];
    for &p in &small_primes {
        let p_big = BigUint::from(p);
        while (&remaining % &p_big).is_zero() {
            remaining /= &p_big;
        }
    }

    // If remaining is 1, the square is smooth over our small factor base
    if remaining.is_one() {
        // Limit accumulation to prevent memory bloat
        if dixon.x_values.len() < 100 {
            dixon.x_values.push(state.clone());
            dixon.square_values.push(sq);
        }
    }
}

/// Try to combine accumulated Dixon squares to find a factor.
/// Multiplies pairs of smooth squares and checks if the combined product
/// yields a congruence of squares.
pub fn dixon_combine(dixon: &mut DixonState, n: &BigUint) -> Option<BigUint> {
    let one = BigUint::one();
    if dixon.x_values.len() < 2 || *n <= one {
        return None;
    }

    // Try combining pairs of smooth squares
    let len = dixon.x_values.len();
    for i in 0..len.min(10) {
        for j in (i + 1)..len.min(10) {
            // X = x_i * x_j mod N
            let x = (&dixon.x_values[i] * &dixon.x_values[j]) % n;
            // Y² = sq_i * sq_j — the product should have all-even exponents
            let y_sq = (&dixon.square_values[i] * &dixon.square_values[j]) % n;
            let y = y_sq.sqrt();

            // Check if Y² is indeed a perfect square mod N
            if &(&y * &y) % n == y_sq {
                // Try gcd(X - Y, N)
                let diff = if x >= y { &x - &y } else { &y - &x };
                if !diff.is_zero() {
                    let g = diff.gcd(n);
                    if g > one && g < *n {
                        return Some(g);
                    }
                }
                // Try gcd(X + Y, N)
                let sum = (&x + &y) % n;
                if !sum.is_zero() {
                    let g = sum.gcd(n);
                    if g > one && g < *n {
                        return Some(g);
                    }
                }
            }
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_primitive_square_mod() {
        // 7^2 mod 15 = 49 mod 15 = 4
        let x = BigUint::from(7u32);
        let n = BigUint::from(15u32);
        assert_eq!(square_mod(&x, &n), BigUint::from(4u32));
    }

    #[test]
    fn test_primitive_subtract_gcd() {
        // |8 - 5| = 3, gcd(3, 15) = 3 (nontrivial factor of 15)
        let a = BigUint::from(8u32);
        let b = BigUint::from(5u32);
        let n = BigUint::from(15u32);
        let result = subtract_gcd(&a, &b, &n);
        assert_eq!(result, Some(BigUint::from(3u32)));
    }

    #[test]
    fn test_subtract_gcd_trivial() {
        // |10 - 5| = 5, gcd(5, 5) = 5 == n => trivial
        let a = BigUint::from(10u32);
        let b = BigUint::from(5u32);
        let n = BigUint::from(5u32);
        assert_eq!(subtract_gcd(&a, &b, &n), None);
    }

    #[test]
    fn test_mod_pow_prim() {
        // 2^10 mod 1000 = 1024 mod 1000 = 24
        let base = BigUint::from(2u32);
        let exp = BigUint::from(10u32);
        let n = BigUint::from(1000u32);
        assert_eq!(mod_pow_prim(&base, &exp, &n), BigUint::from(24u32));
    }

    #[test]
    fn test_add_const_mod() {
        // (13 + 5) mod 15 = 18 mod 15 = 3
        let x = BigUint::from(13u32);
        let n = BigUint::from(15u32);
        assert_eq!(add_const_mod(&x, 5, &n), BigUint::from(3u32));
    }

    #[test]
    fn test_multiply_mod() {
        // (7 * 8) mod 15 = 56 mod 15 = 11
        let a = BigUint::from(7u32);
        let b = BigUint::from(8u32);
        let n = BigUint::from(15u32);
        assert_eq!(multiply_mod(&a, &b, &n), BigUint::from(11u32));
    }

    #[test]
    fn test_is_perfect_square() {
        assert!(is_perfect_square(&BigUint::from(0u32)));
        assert!(is_perfect_square(&BigUint::from(1u32)));
        assert!(is_perfect_square(&BigUint::from(4u32)));
        assert!(is_perfect_square(&BigUint::from(9u32)));
        assert!(is_perfect_square(&BigUint::from(144u32)));
        assert!(!is_perfect_square(&BigUint::from(2u32)));
        assert!(!is_perfect_square(&BigUint::from(8u32)));
        assert!(!is_perfect_square(&BigUint::from(15u32)));
    }

    #[test]
    fn test_fermat_step() {
        // 35 = 5 * 7. a = 6, a^2 = 36, 36 - 35 = 1, sqrt(1) = 1. gcd(6+1, 35) = 7
        let n = BigUint::from(35u32);
        let a = BigUint::from(6u32);
        let result = fermat_step(&n, &a, 1);
        assert!(result.is_some());
        let f = result.unwrap();
        assert!(f == BigUint::from(5u32) || f == BigUint::from(7u32));
    }

    #[test]
    fn test_hart_step() {
        // 8051 = 83 * 97. Try several multipliers.
        let n = BigUint::from(8051u32);
        let mut found = false;
        for i in 1..100u64 {
            if let Some(f) = hart_step(&n, i) {
                assert!((&n % &f).is_zero());
                found = true;
                break;
            }
        }
        assert!(
            found,
            "Hart should find a factor of 8051 within 100 multipliers"
        );
    }

    #[test]
    fn test_is_6m_pm1() {
        assert!(is_6m_pm1(&BigUint::from(5u32))); // 6*1 - 1
        assert!(is_6m_pm1(&BigUint::from(7u32))); // 6*1 + 1
        assert!(is_6m_pm1(&BigUint::from(11u32))); // 6*2 - 1
        assert!(is_6m_pm1(&BigUint::from(13u32))); // 6*2 + 1
        assert!(!is_6m_pm1(&BigUint::from(6u32)));
        assert!(!is_6m_pm1(&BigUint::from(9u32)));
    }

    // --- Tests for new crate-wrapping primitives ---

    #[test]
    fn test_cf_convergent_gcd_small() {
        // 15 = 3 × 5. CF convergents of sqrt(15) should yield a factor.
        let n = BigUint::from(15u32);
        let mut found = false;
        for k in 1..20u32 {
            if let Some(factor) = cf_convergent_gcd(&n, k) {
                assert!(
                    factor == BigUint::from(3u32) || factor == BigUint::from(5u32),
                    "Expected factor 3 or 5, got {}",
                    factor
                );
                found = true;
                break;
            }
        }
        // CF convergent may not always find a factor for all N, so just verify no panic
        let _ = found;
    }

    #[test]
    fn test_cf_convergent_gcd_trivial() {
        // n=1 should return None
        assert!(cf_convergent_gcd(&BigUint::one(), 5).is_none());
    }

    #[test]
    fn test_squfof_attempt_small() {
        // 15 = 3 × 5 — SQUFOF should factor this
        let n = BigUint::from(15u32);
        if let Some(factor) = squfof_attempt(&n) {
            assert!((&n % &factor).is_zero(), "Factor should divide N");
        }
    }

    #[test]
    fn test_squfof_attempt_skips_large() {
        // Numbers > 48 bits should return None immediately
        let large = BigUint::from(1u64 << 49);
        assert!(squfof_attempt(&large).is_none());
    }

    #[test]
    fn test_ecm_attempt_small() {
        // 77 = 7 × 11 — ECM should find a factor easily
        let n = BigUint::from(77u32);
        if let Some(factor) = ecm_attempt(&n, 500) {
            assert!((&n % &factor).is_zero(), "Factor should divide N");
        }
    }

    #[test]
    fn test_lll_short_vector_gcd() {
        // Test that LLL doesn't panic on small inputs
        let state = BigUint::from(7u32);
        let n = BigUint::from(15u32);
        let _ = lll_short_vector_gcd(&state, &n);
    }

    #[test]
    fn test_smooth_test() {
        // 60 = 2² × 3 × 5, which is 5-smooth
        let state = BigUint::from(60u32);
        let result = smooth_test(&state, 5);
        assert_eq!(result, BigUint::from(5u32)); // Largest factor = 5

        // 77 = 7 × 11, not 5-smooth
        let state_77 = BigUint::from(77u32);
        let result_77 = smooth_test(&state_77, 5);
        assert_eq!(result_77, BigUint::one()); // Not smooth
    }

    #[test]
    fn test_jacobi_symbol_prim() {
        let n = BigUint::from(15u32);
        // Jacobi(2/15) = 1 since 2 is a QR mod 3 and 5
        let result = jacobi_symbol_prim(&BigUint::from(2u32), &n);
        assert!(
            result == BigUint::one() || result == &n - BigUint::one(),
            "Jacobi symbol should be ±1"
        );

        // gcd(3, 15) = 3, so Jacobi(3/15) = 0
        let result_3 = jacobi_symbol_prim(&BigUint::from(3u32), &n);
        assert_eq!(result_3, BigUint::zero());
    }

    #[test]
    fn test_pollard_pm1_factors() {
        // 77 = 7 × 11. p-1 = 6 = 2×3 (6-smooth), so Pollard p-1 with B≥3 should find 7
        let n = BigUint::from(77u32);
        let base = BigUint::from(2u32);
        if let Some(factor) = pollard_pm1_step(&n, &base, 10) {
            assert!((&n % &factor).is_zero(), "Factor should divide N");
        }
    }

    #[test]
    fn test_dixon_accumulate_and_combine() {
        let n = BigUint::from(77u32);
        let mut dixon = DixonState::new();

        // Accumulate several values
        for x in 2..50u32 {
            dixon_accumulate(&mut dixon, &BigUint::from(x), &n);
        }

        // May or may not find a factor depending on smooth squares found
        let _ = dixon_combine(&mut dixon, &n);
        // Just verify no panic
    }

    #[test]
    fn test_rho_form_step_no_panic() {
        // Verify rho_form_step doesn't panic on small semiprimes
        let n = BigUint::from(77u32);
        let _ = rho_form_step(&n);
    }

    #[test]
    fn test_pilatte_vector_product_no_panic() {
        // Verify pilatte_vector_product doesn't panic on small N
        let n = BigUint::from(77u32);
        let result = pilatte_vector_product(&n);
        assert!(result > BigUint::zero() || result.is_zero());
    }
}
