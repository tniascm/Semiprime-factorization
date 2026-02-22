//! Group-theoretic analysis of Z/nZ for factorization.
//!
//! Computes algebraic properties of the multiplicative group (Z/nZ)*
//! and explores whether partial group structure reveals factor information.

pub mod chebotarev;

use std::collections::HashMap;

use factoring_core::{gcd, mod_pow};
use num_bigint::BigUint;
use num_traits::{One, Zero};

/// Compute the order of element a in (Z/nZ)*.
pub fn element_order(a: &BigUint, n: &BigUint, max_order: u64) -> Option<u64> {
    let one = BigUint::one();
    let mut current = a % n;
    if current.is_zero() || gcd(&current, n) != one {
        return None; // Not in the group
    }

    for ord in 1..=max_order {
        if current == one {
            return Some(ord);
        }
        current = (&current * a) % n;
    }
    None
}

/// Sample random elements and compute their orders.
/// The distribution of orders reveals group structure.
pub fn sample_orders(n: &BigUint, num_samples: usize, max_order: u64) -> Vec<u64> {
    let mut rng = rand::thread_rng();
    let one = BigUint::one();

    (0..num_samples)
        .filter_map(|_| {
            let bytes = n.to_bytes_be();
            let mut random_bytes = vec![0u8; bytes.len()];
            rand::Rng::fill(&mut rng, &mut random_bytes[..]);
            let a = BigUint::from_bytes_be(&random_bytes) % n;
            if a > one && gcd(&a, n) == one {
                element_order(&a, n, max_order)
            } else {
                None
            }
        })
        .collect()
}

/// Compute Euler's totient function φ(n) when factorization is known.
/// φ(pq) = (p-1)(q-1) for distinct primes p, q.
pub fn euler_totient(p: &BigUint, q: &BigUint) -> BigUint {
    let one = BigUint::one();
    (p - &one) * (q - &one)
}

/// Attempt to discover φ(n) from sampled element orders.
/// If we find φ(n), we can factor n using:
/// p + q = n - φ(n) + 1
/// p - q = sqrt((p+q)^2 - 4n)
pub fn phi_from_orders(n: &BigUint, orders: &[u64]) -> Option<BigUint> {
    if orders.is_empty() {
        return None;
    }

    // φ(n) must be divisible by all element orders
    // The LCM of all observed orders approximates λ(n) (Carmichael function)
    // For RSA numbers: λ(n) = lcm(p-1, q-1)
    // And φ(n) = (p-1)(q-1) is a multiple of λ(n)

    let mut lcm_val = 1u64;
    for &ord in orders {
        lcm_val = lcm(lcm_val, ord);
        if lcm_val > 1_000_000_000 {
            break; // Overflow protection
        }
    }

    // lcm_val approximates λ(n)
    // Try multiples of λ(n) as candidates for φ(n)
    let one = BigUint::one();
    for mult in 1..=100u64 {
        let phi_candidate = BigUint::from(lcm_val * mult);
        // Check: a^phi ≡ 1 (mod n) for several random a
        // This is necessary but not sufficient
        let a = BigUint::from(2u32);
        if mod_pow(&a, &phi_candidate, n) == one {
            let a = BigUint::from(3u32);
            if mod_pow(&a, &phi_candidate, n) == one {
                return Some(phi_candidate);
            }
        }
    }

    None
}

/// Factor n given φ(n).
pub fn factor_from_phi(n: &BigUint, phi: &BigUint) -> Option<(BigUint, BigUint)> {
    let one = BigUint::one();
    // p + q = n - φ(n) + 1
    let sum = n - phi + &one;
    // p * q = n
    // p and q are roots of: x^2 - (p+q)x + n = 0
    // discriminant = (p+q)^2 - 4n
    let sum_sq = &sum * &sum;
    let four_n = BigUint::from(4u32) * n;
    if sum_sq < four_n {
        return None;
    }
    let disc = &sum_sq - &four_n;
    let sqrt_disc = disc.sqrt();
    if &sqrt_disc * &sqrt_disc != disc {
        return None; // Not a perfect square — φ(n) was wrong
    }

    let two = BigUint::from(2u32);
    let p = (&sum + &sqrt_disc) / &two;
    let q = (&sum - &sqrt_disc) / &two;

    if &p * &q == *n {
        Some((p, q))
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Pohlig-Hellman order decomposition
// ---------------------------------------------------------------------------

/// Factor an integer into (prime, exponent) pairs via trial division.
pub fn factor_order(order: u64) -> Vec<(u64, u32)> {
    let mut factors: Vec<(u64, u32)> = Vec::new();
    let mut remaining = order;

    if remaining <= 1 {
        return factors;
    }

    // Factor out 2
    let mut exp = 0u32;
    while remaining % 2 == 0 {
        exp += 1;
        remaining /= 2;
    }
    if exp > 0 {
        factors.push((2, exp));
    }

    // Odd trial divisors
    let mut d = 3u64;
    while d.saturating_mul(d) <= remaining {
        let mut exp = 0u32;
        while remaining % d == 0 {
            exp += 1;
            remaining /= d;
        }
        if exp > 0 {
            factors.push((d, exp));
        }
        d += 2;
    }

    if remaining > 1 {
        factors.push((remaining, 1));
    }

    factors
}

/// Compute the order of `a` modulo `n` using the Pohlig-Hellman decomposition.
///
/// Given an upper bound on the order (`order_bound`), we factor the bound,
/// find the order component in each prime-power subgroup, and combine via
/// the Chinese Remainder Theorem to get the full order.
///
/// Falls back to sequential search when `order_bound` is small (< 256).
pub fn pohlig_hellman_order(a: &BigUint, n: &BigUint, order_bound: u64) -> Option<u64> {
    let one = BigUint::one();

    // Not in the group check
    let a_mod = a % n;
    if a_mod.is_zero() || gcd(&a_mod, n) != one {
        return None;
    }

    // For small bounds just do sequential search
    if order_bound < 256 {
        return element_order(a, n, order_bound);
    }

    // Factor the order bound
    let factors = factor_order(order_bound);
    if factors.is_empty() {
        // order_bound was 0 or 1, so a must be 1
        return if a_mod == one { Some(1) } else { None };
    }

    // For each prime power p^e dividing the bound, find the order of a
    // in the subgroup of size p^e.
    //
    // The idea: let m = order_bound. Raise a to m / p^e. The result has
    // order dividing p^e. Then find the exact order by successive division.
    let mut order: u64 = 1;

    for &(p, e) in &factors {
        let prime_power = p.checked_pow(e).unwrap_or(u64::MAX);

        // cofactor = order_bound / p^e
        let cofactor = order_bound / prime_power;
        // a_sub = a^cofactor mod n  — lives in the p-subgroup
        let a_sub = mod_pow(&a_mod, &BigUint::from(cofactor), n);

        // Find the smallest k in 0..=e such that a_sub^(p^k) ≡ 1 (mod n)
        let mut sub_order: u64 = 1;
        let mut powered = a_sub.clone();
        for _k in 0..e {
            if powered == one {
                break;
            }
            sub_order *= p;
            powered = mod_pow(&powered, &BigUint::from(p), n);
        }

        order = lcm(order, sub_order);
    }

    // Verify: a^order ≡ 1 (mod n)
    if mod_pow(&a_mod, &BigUint::from(order), n) == one {
        Some(order)
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Baby-step giant-step
// ---------------------------------------------------------------------------

/// Solve the discrete logarithm base^x ≡ target (mod n) for 0 <= x < order
/// using the baby-step giant-step algorithm.
///
/// Time and space complexity: O(sqrt(order)).
pub fn baby_step_giant_step(
    base: &BigUint,
    target: &BigUint,
    n: &BigUint,
    order: u64,
) -> Option<u64> {
    let one = BigUint::one();

    // m = ceil(sqrt(order))
    let m = (order as f64).sqrt().ceil() as u64;
    if m == 0 {
        // order is 0 — only solution is if target ≡ 1
        return if (target % n) == one { Some(0) } else { None };
    }

    // Baby step: build table of base^j mod n for j = 0..m-1
    let mut table: HashMap<Vec<u8>, u64> = HashMap::with_capacity(m as usize);
    let mut power = one.clone();
    for j in 0..m {
        let key = (&power % n).to_bytes_be();
        table.entry(key).or_insert(j);
        power = (&power * base) % n;
    }

    // Giant step factor: base^(-m) mod n
    // base^(order) ≡ 1 (mod n), so base^(-m) ≡ base^(order - m % order) (mod n)
    let exp_inv = {
        let m_mod_order = m % order;
        if m_mod_order == 0 {
            0u64
        } else {
            order - m_mod_order
        }
    };
    let base_m_inv = mod_pow(base, &BigUint::from(exp_inv), n);

    // Giant step: for i = 0..m, check if target * (base^(-m))^i is in the table
    let mut gamma = target % n;
    for i in 0..m {
        let key = gamma.to_bytes_be();
        if let Some(&j) = table.get(&key) {
            let x = i * m + j;
            if x < order {
                // Verify
                if mod_pow(base, &BigUint::from(x), n) == (target % n) {
                    return Some(x);
                }
            }
        }
        gamma = (&gamma * &base_m_inv) % n;
    }

    None
}

// ---------------------------------------------------------------------------
// Smooth order detection
// ---------------------------------------------------------------------------

/// Check whether `value` is B-smooth (all prime factors <= `smoothness_bound`).
fn is_smooth(value: u64, smoothness_bound: u64) -> bool {
    let factors = factor_order(value);
    factors.iter().all(|&(p, _)| p <= smoothness_bound)
}

/// Sample random elements of (Z/nZ)* and return the first one whose
/// order is B-smooth (all prime factors at most `smoothness_bound`).
///
/// Returns `(element, order)` on success.
pub fn find_smooth_order_element(
    n: &BigUint,
    smoothness_bound: u64,
    num_trials: usize,
) -> Option<(BigUint, u64)> {
    let mut rng = rand::thread_rng();
    let one = BigUint::one();

    // Determine a reasonable max_order for searching.
    // For small n we can search all the way; otherwise cap at a workable limit.
    let n_u64: u64 = n.to_u64_digits().first().copied().unwrap_or(u64::MAX);
    let max_order = n_u64.min(100_000);

    for _ in 0..num_trials {
        let bytes = n.to_bytes_be();
        let mut random_bytes = vec![0u8; bytes.len()];
        rand::Rng::fill(&mut rng, &mut random_bytes[..]);
        let a = BigUint::from_bytes_be(&random_bytes) % n;
        if a <= one || gcd(&a, n) != one {
            continue;
        }

        if let Some(ord) = element_order(&a, n, max_order) {
            if is_smooth(ord, smoothness_bound) {
                return Some((a, ord));
            }
        }
    }

    None
}

/// Attempt to factor `n` by finding elements with smooth orders.
///
/// Strategy: collect several element orders, compute their LCM (an
/// approximation of λ(n)), then try small multiples as φ(n) candidates
/// and use `factor_from_phi` to recover the factors.
pub fn factor_via_smooth_orders(
    n: &BigUint,
    smoothness_bound: u64,
    num_trials: usize,
) -> Option<BigUint> {
    let one = BigUint::one();
    let mut lambda_approx: u64 = 1;

    let mut rng = rand::thread_rng();
    let n_u64: u64 = n.to_u64_digits().first().copied().unwrap_or(u64::MAX);
    let max_order = n_u64.min(100_000);

    // Collect smooth orders and build up LCM
    let mut smooth_count = 0usize;
    for _ in 0..num_trials {
        let bytes = n.to_bytes_be();
        let mut random_bytes = vec![0u8; bytes.len()];
        rand::Rng::fill(&mut rng, &mut random_bytes[..]);
        let a = BigUint::from_bytes_be(&random_bytes) % n;
        if a <= one || gcd(&a, n) != one {
            continue;
        }

        if let Some(ord) = element_order(&a, n, max_order) {
            if is_smooth(ord, smoothness_bound) {
                lambda_approx = lcm(lambda_approx, ord);
                smooth_count += 1;
                if lambda_approx > 1_000_000_000 {
                    break; // Overflow guard
                }
            }
        }
    }

    if smooth_count == 0 || lambda_approx <= 1 {
        return None;
    }

    // Try multiples of lambda_approx as phi(n) candidates
    for mult in 1..=200u64 {
        let phi_candidate = BigUint::from(lambda_approx.saturating_mul(mult));
        // Quick plausibility: 2^phi ≡ 1 (mod n) and 3^phi ≡ 1 (mod n)
        if mod_pow(&BigUint::from(2u32), &phi_candidate, n) != one {
            continue;
        }
        if mod_pow(&BigUint::from(3u32), &phi_candidate, n) != one {
            continue;
        }
        if let Some((p, _q)) = factor_from_phi(n, &phi_candidate) {
            return Some(p);
        }
    }

    None
}

// ---------------------------------------------------------------------------
// Carmichael function approximation
// ---------------------------------------------------------------------------

/// Approximate the Carmichael function λ(n) by sampling many element orders
/// and computing their LCM.
///
/// For n = pq (distinct primes), λ(n) = lcm(p-1, q-1).
/// If sufficient elements are sampled, the LCM of their orders converges
/// to λ(n).
///
/// Returns the approximated λ(n) value, or `None` if sampling fails.
pub fn approximate_carmichael(
    n: &BigUint,
    num_samples: usize,
    max_order: u64,
) -> Option<u64> {
    let orders = sample_orders(n, num_samples, max_order);
    if orders.is_empty() {
        return None;
    }

    let mut lambda: u64 = 1;
    for &ord in &orders {
        lambda = lcm(lambda, ord);
        if lambda > 1_000_000_000 {
            return Some(lambda); // Return what we have, even if saturated
        }
    }

    if lambda > 1 {
        Some(lambda)
    } else {
        None
    }
}

fn lcm(a: u64, b: u64) -> u64 {
    if a == 0 || b == 0 {
        return 0;
    }
    a / gcd_u64(a, b) * b
}

fn gcd_u64(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_element_order() {
        let n = BigUint::from(15u32);
        let a = BigUint::from(2u32);
        // Order of 2 in (Z/15Z)* is 4
        let ord = element_order(&a, &n, 100);
        assert_eq!(ord, Some(4));
    }

    #[test]
    fn test_euler_totient() {
        let p = BigUint::from(5u32);
        let q = BigUint::from(7u32);
        assert_eq!(euler_totient(&p, &q), BigUint::from(24u32));
    }

    #[test]
    fn test_factor_from_phi() {
        let n = BigUint::from(35u32);
        let phi = BigUint::from(24u32);
        let result = factor_from_phi(&n, &phi);
        assert!(result.is_some());
        let (p, q) = result.unwrap();
        assert_eq!(&p * &q, n);
    }

    #[test]
    fn test_factor_order() {
        // 12 = 2^2 * 3
        let factors = factor_order(12);
        assert_eq!(factors, vec![(2, 2), (3, 1)]);

        // 1 has no prime factors
        assert!(factor_order(1).is_empty());

        // Prime number
        assert_eq!(factor_order(17), vec![(17, 1)]);

        // Power of 2
        assert_eq!(factor_order(8), vec![(2, 3)]);

        // Larger composite
        // 360 = 2^3 * 3^2 * 5
        let factors = factor_order(360);
        assert_eq!(factors, vec![(2, 3), (3, 2), (5, 1)]);
    }

    #[test]
    fn test_baby_step_giant_step() {
        let n = BigUint::from(15u32);

        // 2^x ≡ 8 (mod 15) → x = 3  (since 2^3 = 8)
        let base = BigUint::from(2u32);
        let target = BigUint::from(8u32);
        // Order of 2 mod 15 is 4
        let result = baby_step_giant_step(&base, &target, &n, 4);
        assert_eq!(result, Some(3));

        // 2^0 ≡ 1 (mod 15) → x = 0
        let target_one = BigUint::from(1u32);
        let result = baby_step_giant_step(&base, &target_one, &n, 4);
        assert_eq!(result, Some(0));

        // 2^1 ≡ 2 (mod 15) → x = 1
        let target_two = BigUint::from(2u32);
        let result = baby_step_giant_step(&base, &target_two, &n, 4);
        assert_eq!(result, Some(1));

        // 2^2 ≡ 4 (mod 15) → x = 2
        let target_four = BigUint::from(4u32);
        let result = baby_step_giant_step(&base, &target_four, &n, 4);
        assert_eq!(result, Some(2));

        // No solution: 3 is not a power of 2 mod 15 within order 4
        // 2^0=1, 2^1=2, 2^2=4, 2^3=8 mod 15
        let target_three = BigUint::from(3u32);
        let result = baby_step_giant_step(&base, &target_three, &n, 4);
        assert_eq!(result, None);
    }

    #[test]
    fn test_pohlig_hellman_order() {
        let n = BigUint::from(35u32);

        // Order of 2 mod 35: 2^1=2, 2^2=4, 2^3=8, 2^4=16, 2^5=32,
        // 2^6=64%35=29, 2^7=58%35=23, 2^8=46%35=11, 2^9=22, 2^10=44%35=9,
        // 2^11=18, 2^12=36%35=1  → order = 12
        let a = BigUint::from(2u32);
        let result = pohlig_hellman_order(&a, &n, 24);
        assert_eq!(result, Some(12));

        // Verify sequential search gives the same answer
        let sequential = element_order(&a, &n, 100);
        assert_eq!(sequential, result);

        // Test with a small bound that triggers the fallback
        let a3 = BigUint::from(3u32);
        let result_small = pohlig_hellman_order(&a3, &n, 100);
        let sequential_small = element_order(&a3, &n, 100);
        assert_eq!(result_small, sequential_small);
    }

    #[test]
    fn test_smooth_order_detection() {
        // n = 35 = 5 * 7
        // φ(35) = 24, λ(35) = lcm(4, 6) = 12
        // Possible orders: divisors of 12 = {1, 2, 3, 4, 6, 12}
        // All of these are 7-smooth (prime factors are at most 3 ≤ 7)
        let n = BigUint::from(35u32);

        // With a generous smoothness bound, we should find smooth elements easily
        let result = find_smooth_order_element(&n, 7, 200);
        assert!(
            result.is_some(),
            "Should find a smooth-order element in Z/35Z"
        );

        let (elem, ord) = result.unwrap();
        // Verify the element actually has this order
        assert_eq!(
            mod_pow(&elem, &BigUint::from(ord), &n),
            BigUint::one(),
            "Element raised to its order should be 1"
        );

        // Verify smoothness
        assert!(
            is_smooth(ord, 7),
            "Order {} should be 7-smooth",
            ord
        );
    }

    #[test]
    fn test_carmichael_approximation() {
        // n = 35 = 5 * 7
        // λ(35) = lcm(4, 6) = 12
        let n = BigUint::from(35u32);

        // With enough samples, the LCM of orders should be 12
        // (or a multiple, but for this small n, it converges quickly)
        let result = approximate_carmichael(&n, 200, 100);
        assert!(result.is_some(), "Should approximate λ(35)");

        let lambda = result.unwrap();
        // λ(35) = 12. The approximation must be a multiple of the true value
        // if we've sampled generators for both prime subgroups. For n=35
        // with 200 samples, we almost certainly get exactly 12.
        assert_eq!(
            lambda % 12,
            0,
            "Approximated λ(n) = {} should be divisible by 12",
            lambda
        );

        // The value should divide or equal 12 for n=35 since all orders
        // divide 12 and 12 itself is achievable.
        assert_eq!(
            lambda, 12,
            "For n=35 with sufficient samples, λ should be exactly 12"
        );
    }
}
