//! Sublinear conductor detection via sampling.
//!
//! Instead of enumerating all phi(N) characters, we try to find a character
//! with conductor < N using various sampling strategies.
//!
//! For N = pq, exactly (p-1) characters have conductor q and (q-1) have
//! conductor p. A random character has probability ~1/min(p,q) of revealing
//! a factor. For balanced semiprimes, ~1/sqrt(N).

use crate::characters::{mod_pow, multiplicative_group_generators};
use crate::complex;
use num_integer::Integer;
use rand::Rng;
use std::f64::consts::PI;
use std::time::Instant;

/// Result of a sampling-based conductor detection experiment.
#[derive(Debug, Clone)]
pub struct SamplingResult {
    /// The number that was tested.
    pub n: u64,
    /// The name of the sampling method used.
    pub method: String,
    /// The number of samples tested before termination.
    pub samples_tested: u64,
    /// The factor found, if any, as (p, q) with p * q = n.
    pub factor_found: Option<(u64, u64)>,
    /// Wall-clock time in microseconds.
    pub time_us: u64,
    /// Diagnostic details about the run.
    pub details: Vec<String>,
}

/// Method 1: Random character sampling baseline.
///
/// Sample random elements of (Z/NZ)*, construct characters from them,
/// and check conductors.
///
/// Expected samples needed: O(min(p,q)) for N=pq balanced = O(sqrt(N)) = exponential.
/// This establishes the baseline that smarter methods must beat.
///
/// In practice, we use a Pollard p-1 style approach: pick random a, compute
/// a^(product of small primes) mod N. If a^B - 1 shares a non-trivial gcd
/// with N, we have found a factor.
pub fn random_sampling(n: u64, max_samples: u64) -> SamplingResult {
    let start = Instant::now();
    let mut rng = rand::thread_rng();
    let mut details = Vec::new();

    for sample in 0..max_samples {
        let a = rng.gen_range(2..n);
        if a.gcd(&n) > 1 {
            let g = a.gcd(&n);
            if g < n {
                details.push(format!("Lucky: gcd({}, {}) = {}", a, n, g));
                return SamplingResult {
                    n,
                    method: "random_sampling".into(),
                    samples_tested: sample + 1,
                    factor_found: Some((g, n / g)),
                    time_us: start.elapsed().as_micros() as u64,
                    details,
                };
            }
        }

        // Compute a^(product of small primes) mod N (Pollard p-1 style)
        let mut power = a;
        for &p in &[2u64, 2, 2, 2, 3, 3, 5, 7, 11, 13, 17, 19, 23] {
            power = mod_pow(power, p, n);
        }
        let diff = if power >= 1 { power - 1 } else { n - 1 };
        let g = diff.gcd(&n);
        if g > 1 && g < n {
            details.push(format!(
                "p-1 style: found factor {} at sample {}",
                g, sample
            ));
            return SamplingResult {
                n,
                method: "random_sampling".into(),
                samples_tested: sample + 1,
                factor_found: Some((g, n / g)),
                time_us: start.elapsed().as_micros() as u64,
                details,
            };
        }
    }

    SamplingResult {
        n,
        method: "random_sampling".into(),
        samples_tested: max_samples,
        factor_found: None,
        time_us: start.elapsed().as_micros() as u64,
        details,
    }
}

/// Method 2: Structured sampling via subgroup chains.
///
/// Characters of small order are more likely to have small conductor.
/// Enumerate characters of order dividing d for small d.
///
/// Key insight: For N = pq, a character chi of order d has conductor dividing N.
/// If d | (p-1) but d does not divide (q-1), then any character of exact order d
/// has conductor p (or dividing p*something, revealing p via gcd).
///
/// So: find d that divides (p-1) but not (q-1), then any character of exact order d
/// has conductor p (or dividing p*something, revealing p via gcd).
///
/// In practice: for each small d, test a^d mod N for random a. If a^d = 1 (mod p)
/// but a^d != 1 (mod q), then gcd(a^d - 1, N) = p.
pub fn subgroup_chain_sampling(n: u64, max_order: u64) -> SamplingResult {
    let start = Instant::now();
    let mut details = Vec::new();

    let mut rng = rand::thread_rng();
    let mut samples = 0u64;

    for d in 2..=max_order {
        // Try several random bases for each order
        for _ in 0..20 {
            samples += 1;
            let a = rng.gen_range(2..n);
            if a.gcd(&n) > 1 {
                let g = a.gcd(&n);
                if g < n {
                    return SamplingResult {
                        n,
                        method: "subgroup_chain".into(),
                        samples_tested: samples,
                        factor_found: Some((g, n / g)),
                        time_us: start.elapsed().as_micros() as u64,
                        details,
                    };
                }
                continue;
            }

            let power = mod_pow(a, d, n);
            if power == 0 {
                continue;
            }
            let diff = if power >= 1 { power - 1 } else { n - 1 };
            let g = diff.gcd(&n);
            if g > 1 && g < n {
                details.push(format!(
                    "Order {}: a={}, a^{}={} mod {}, gcd={}",
                    d, a, d, power, n, g
                ));
                return SamplingResult {
                    n,
                    method: "subgroup_chain".into(),
                    samples_tested: samples,
                    factor_found: Some((g, n / g)),
                    time_us: start.elapsed().as_micros() as u64,
                    details,
                };
            }
        }
    }

    SamplingResult {
        n,
        method: "subgroup_chain".into(),
        samples_tested: samples,
        factor_found: None,
        time_us: start.elapsed().as_micros() as u64,
        details,
    }
}

/// Method 3: Partial Gauss sum estimation.
///
/// Instead of computing the full Gauss sum tau(chi) = sum_{a=0}^{N-1} chi(a) * e^{2pi i a/N},
/// estimate it from O(sqrt(N)) or fewer random terms.
///
/// For a primitive character mod N: |tau(chi)|^2 = N.
/// For a character with conductor f < N: |tau(chi)|^2 != N.
///
/// We compute tau_partial = sum of k random terms from the Gauss sum,
/// and check if |tau_partial|^2 * (N/k) is close to N or not.
pub fn partial_gauss_estimation(
    n: u64,
    num_partial_terms: u64,
    num_trials: u64,
) -> SamplingResult {
    let start = Instant::now();
    let mut rng = rand::thread_rng();
    let mut details = Vec::new();

    // We need actual characters. For this method, generate a random character
    // by choosing random images of generators.
    let generators = multiplicative_group_generators(n);
    if generators.is_empty() {
        return SamplingResult {
            n,
            method: "partial_gauss".into(),
            samples_tested: 0,
            factor_found: None,
            time_us: start.elapsed().as_micros() as u64,
            details: vec!["No generators found".into()],
        };
    }

    let orders: Vec<u64> = generators.iter().map(|g| g.order).collect();

    for trial in 0..num_trials {
        // Generate random character by choosing random exponents for each generator
        let exponents: Vec<u64> = orders.iter().map(|&ord| rng.gen_range(0..ord)).collect();

        // Check if this is the principal character (all exponents 0)
        if exponents.iter().all(|&e| e == 0) {
            continue;
        }

        // Compute partial Gauss sum from random sample of terms
        let mut partial_sum = (0.0f64, 0.0f64);
        let mut terms_computed = 0u64;

        for _ in 0..num_partial_terms {
            let a = rng.gen_range(1..n);
            if a.gcd(&n) != 1 {
                continue;
            }

            // Skip if N is too large for the discrete log approach
            if n > 100000 {
                continue;
            }

            // Compute chi(a) by finding discrete logs w.r.t. each generator
            let mut chi_a = (1.0f64, 0.0f64);
            let mut found_all = true;
            for (i, gen) in generators.iter().enumerate() {
                // Find discrete log of a w.r.t generator i
                let mut power = 1u64;
                let mut found = false;
                for k in 0..gen.order {
                    if power == a % n {
                        let angle = 2.0 * PI * (exponents[i] as f64) * (k as f64)
                            / (orders[i] as f64);
                        let rot = (angle.cos(), angle.sin());
                        chi_a = complex::cmul(chi_a, rot);
                        found = true;
                        break;
                    }
                    power = ((power as u128 * gen.generator as u128) % n as u128) as u64;
                }
                if !found {
                    chi_a = (0.0, 0.0);
                    found_all = false;
                    break;
                }
            }

            if !found_all || complex::cnorm_sq(chi_a) < 1e-10 {
                continue;
            }

            let angle = 2.0 * PI * (a as f64) / (n as f64);
            let exp_term = (angle.cos(), angle.sin());
            let term = complex::cmul(chi_a, exp_term);
            partial_sum = complex::cadd(partial_sum, term);
            terms_computed += 1;
        }

        if terms_computed < 5 {
            continue;
        }

        // Estimate |tau(chi)|^2 by scaling the partial sum
        let scale = (n as f64) / (terms_computed as f64);
        let estimated_mag_sq = complex::cnorm_sq(partial_sum) * scale;

        // For a primitive character, |tau|^2 = N
        // For conductor f < N, |tau|^2 is related to f
        // If estimated_mag_sq is significantly different from N, character may be imprimitive
        let ratio = estimated_mag_sq / (n as f64);

        if ratio < 0.5 || ratio > 2.0 {
            details.push(format!(
                "Trial {}: ratio={:.3}, may be imprimitive",
                trial, ratio
            ));
        }
    }

    SamplingResult {
        n,
        method: "partial_gauss".into(),
        samples_tested: num_trials,
        factor_found: None,
        time_us: start.elapsed().as_micros() as u64,
        details,
    }
}

/// Method 4: Cross-correlation conductor detection.
///
/// For two characters chi1, chi2, if they share the same p-component
/// (i.e., chi1|_{Z/pZ}* = chi2|_{Z/pZ}*), then chi1 * chi2^{-1} has conductor
/// dividing q. Detecting such pairs reveals q.
///
/// Strategy: pick random pairs of bases a, b. Compute (a*b^{-1})^d mod N
/// for various d. If we find d such that (a*b^{-1})^d = 1 mod p but not mod q,
/// then gcd((a*b^{-1})^d - 1, N) = p.
pub fn cross_correlation(n: u64, max_samples: u64) -> SamplingResult {
    let start = Instant::now();
    let mut rng = rand::thread_rng();
    let mut details = Vec::new();
    let mut samples = 0u64;

    for _ in 0..max_samples {
        samples += 1;
        let a = rng.gen_range(2..n);
        let b = rng.gen_range(2..n);

        if a.gcd(&n) > 1 {
            let g = a.gcd(&n);
            if g > 1 && g < n {
                return SamplingResult {
                    n,
                    method: "cross_correlation".into(),
                    samples_tested: samples,
                    factor_found: Some((g, n / g)),
                    time_us: start.elapsed().as_micros() as u64,
                    details,
                };
            }
            continue;
        }
        if b.gcd(&n) > 1 {
            let g = b.gcd(&n);
            if g > 1 && g < n {
                return SamplingResult {
                    n,
                    method: "cross_correlation".into(),
                    samples_tested: samples,
                    factor_found: Some((g, n / g)),
                    time_us: start.elapsed().as_micros() as u64,
                    details,
                };
            }
            continue;
        }

        // Compute a * b^{-1} mod N using extended Euclidean
        let b_inv = mod_inverse_u64(b, n);
        if b_inv == 0 {
            continue;
        }
        let ratio = ((a as u128 * b_inv as u128) % n as u128) as u64;

        // Try ratio^d for small d values
        // If ratio^d = 1 (mod p) but not (mod q), gcd reveals p
        for &d in &[
            2u64, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 18, 20, 24, 30, 36, 42, 48, 60,
        ] {
            let power = mod_pow(ratio, d, n);
            if power == 1 {
                continue;
            } // trivially 1 mod N, no info
            let diff = if power >= 1 { power - 1 } else { n - 1 };
            let g = diff.gcd(&n);
            if g > 1 && g < n {
                details.push(format!(
                    "ratio={}*{}^-1, d={}, gcd={}",
                    a, b, d, g
                ));
                return SamplingResult {
                    n,
                    method: "cross_correlation".into(),
                    samples_tested: samples,
                    factor_found: Some((g, n / g)),
                    time_us: start.elapsed().as_micros() as u64,
                    details,
                };
            }
        }
    }

    SamplingResult {
        n,
        method: "cross_correlation".into(),
        samples_tested: samples,
        factor_found: None,
        time_us: start.elapsed().as_micros() as u64,
        details,
    }
}

/// Method 5: Conductor witness (Miller-Rabin analogy).
///
/// Miller-Rabin works by testing a^{(n-1)/2^k} for witnesses a.
/// By analogy, for conductor detection, test a^{(N-1)/d} mod N for structured d values.
///
/// Key observation: if d | (p-1) but d does not divide (q-1), then for any a coprime to N:
///   a^{(N-1)/d} mod p can be non-trivial
///   a^{(N-1)/d} mod q has specific behavior
/// The GCD of a^{(N-1)/d} - root_of_unity and N may reveal p.
pub fn conductor_witness(n: u64, max_witnesses: u64) -> SamplingResult {
    let start = Instant::now();
    let mut rng = rand::thread_rng();
    let mut details = Vec::new();
    let mut samples = 0u64;

    // N-1 factored to find divisors to try
    let n_minus_1 = n - 1;
    let small_factors = small_prime_factors(n_minus_1);

    for _ in 0..max_witnesses {
        samples += 1;
        let a = rng.gen_range(2..n);
        if a.gcd(&n) > 1 {
            let g = a.gcd(&n);
            if g > 1 && g < n {
                return SamplingResult {
                    n,
                    method: "conductor_witness".into(),
                    samples_tested: samples,
                    factor_found: Some((g, n / g)),
                    time_us: start.elapsed().as_micros() as u64,
                    details,
                };
            }
            continue;
        }

        // Test a^{(N-1)/d} mod N for each small prime factor d of N-1
        for &d in &small_factors {
            if d == 0 {
                continue;
            }
            let exp = n_minus_1 / d;
            let val = mod_pow(a, exp, n);

            // If val = 1 mod one factor but not the other -> reveals factor
            if val != 1 && val != n - 1 {
                let diff = if val >= 1 { val - 1 } else { n - 1 };
                let g = diff.gcd(&n);
                if g > 1 && g < n {
                    details.push(format!(
                        "Witness a={}, d={}, a^((N-1)/d)={}, gcd={}",
                        a, d, val, g
                    ));
                    return SamplingResult {
                        n,
                        method: "conductor_witness".into(),
                        samples_tested: samples,
                        factor_found: Some((g, n / g)),
                        time_us: start.elapsed().as_micros() as u64,
                        details,
                    };
                }
                // Also try gcd(val + 1, N)
                let g = (val + 1).gcd(&n);
                if g > 1 && g < n {
                    details.push(format!(
                        "Witness a={}, d={}, gcd(val+1, N)={}",
                        a, d, g
                    ));
                    return SamplingResult {
                        n,
                        method: "conductor_witness".into(),
                        samples_tested: samples,
                        factor_found: Some((g, n / g)),
                        time_us: start.elapsed().as_micros() as u64,
                        details,
                    };
                }
            }
        }
    }

    SamplingResult {
        n,
        method: "conductor_witness".into(),
        samples_tested: samples,
        factor_found: None,
        time_us: start.elapsed().as_micros() as u64,
        details,
    }
}

/// Find small prime factors of n (up to bound 1000).
fn small_prime_factors(n: u64) -> Vec<u64> {
    let mut factors = Vec::new();
    let mut temp = n;
    for p in 2..=1000u64 {
        if p * p > temp {
            break;
        }
        if temp % p == 0 {
            factors.push(p);
            while temp % p == 0 {
                temp /= p;
            }
        }
    }
    if temp > 1 {
        factors.push(temp);
    }
    factors
}

/// Modular inverse using extended Euclidean algorithm.
fn mod_inverse_u64(a: u64, m: u64) -> u64 {
    if m == 1 {
        return 0;
    }
    let a = a % m;
    if a == 0 {
        return 0;
    }
    let (mut old_r, mut r) = (a as i128, m as i128);
    let (mut old_s, mut s) = (1i128, 0i128);
    while r != 0 {
        let q = old_r / r;
        let temp_r = r;
        r = old_r - q * r;
        old_r = temp_r;
        let temp_s = s;
        s = old_s - q * s;
        old_s = temp_s;
    }
    ((old_s % m as i128 + m as i128) % m as i128) as u64
}

/// Run all five methods and compare.
pub fn run_all_methods(n: u64, max_samples: u64) -> Vec<SamplingResult> {
    let sqrt_max = (max_samples as f64).sqrt() as u64;
    let gauss_terms = (n as f64).sqrt() as u64;
    let gauss_trials = if max_samples >= 10 {
        max_samples / 10
    } else {
        1
    };

    vec![
        random_sampling(n, max_samples),
        subgroup_chain_sampling(n, sqrt_max),
        partial_gauss_estimation(n, gauss_terms, gauss_trials),
        cross_correlation(n, max_samples),
        conductor_witness(n, max_samples),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_random_sampling_small() {
        // Should factor 77 = 7 * 11 quickly
        let result = random_sampling(77, 1000);
        assert!(result.factor_found.is_some(), "Should factor 77");
        let (a, b) = result.factor_found.unwrap();
        assert_eq!(a * b, 77);
    }

    #[test]
    fn test_subgroup_chain_small() {
        let result = subgroup_chain_sampling(77, 50);
        assert!(
            result.factor_found.is_some(),
            "Should factor 77 via subgroup chain"
        );
        let (a, b) = result.factor_found.unwrap();
        assert_eq!(a * b, 77);
    }

    #[test]
    fn test_cross_correlation_small() {
        let result = cross_correlation(77, 1000);
        assert!(
            result.factor_found.is_some(),
            "Should factor 77 via cross-correlation"
        );
        let (a, b) = result.factor_found.unwrap();
        assert_eq!(a * b, 77);
    }

    #[test]
    fn test_conductor_witness_small() {
        let result = conductor_witness(77, 1000);
        assert!(
            result.factor_found.is_some(),
            "Should factor 77 via conductor witness"
        );
        let (a, b) = result.factor_found.unwrap();
        assert_eq!(a * b, 77);
    }

    #[test]
    fn test_all_methods_143() {
        let results = run_all_methods(143, 1000);
        let any_success = results.iter().any(|r| r.factor_found.is_some());
        assert!(any_success, "At least one method should factor 143");
    }

    #[test]
    fn test_scaling_comparison() {
        // Compare how many samples each method needs for increasing N
        let test_cases = [(77u64, 7, 11), (323, 17, 19), (1007, 19, 53)];
        for &(n, _p, _q) in &test_cases {
            let results = run_all_methods(n, 5000);
            for r in &results {
                if r.factor_found.is_some() {
                    // Verify correctness
                    let (a, b) = r.factor_found.unwrap();
                    assert_eq!(a * b, n);
                }
            }
        }
    }

    #[test]
    fn test_mod_inverse() {
        // 3 * 5 = 15 = 1 mod 7
        assert_eq!(mod_inverse_u64(3, 7), 5);
        // 2 * 6 = 12 = 1 mod 11
        assert_eq!(mod_inverse_u64(2, 11), 6);
        // No inverse for 0
        assert_eq!(mod_inverse_u64(0, 7), 0);
    }

    #[test]
    fn test_small_prime_factors() {
        let factors = small_prime_factors(60);
        assert_eq!(factors, vec![2, 3, 5]);

        let factors = small_prime_factors(77);
        assert_eq!(factors, vec![7, 11]);
    }
}
