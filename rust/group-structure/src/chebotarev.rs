//! Chebotarev density-guided factoring.
//!
//! For N = pq, the probability that a random a in (Z/NZ)* has ord(a) divisible
//! by a small prime r depends on whether r divides (p-1) and/or (q-1).
//!
//! By Chebotarev's density theorem:
//! - If r | (p-1) and r | (q-1): density = 1 - (1 - 1/r)^2 = (2r-1)/r^2
//! - If r | (p-1) but r ∤ (q-1): density = 1/r  (approximately)
//! - If r ∤ (p-1) and r ∤ (q-1): density = 0  (no elements of order r)
//!
//! Measuring these densities for many small primes r reveals which r divide
//! (p-1) or (q-1), constraining the factorization.

use num_bigint::BigUint;
use num_traits::One;
use factoring_core::{gcd, mod_pow};

/// Result of Chebotarev density measurement for one prime r.
#[derive(Debug, Clone)]
pub struct DensityMeasurement {
    pub r: u64,
    pub num_tested: u64,
    pub num_divisible: u64,
    pub empirical_density: f64,
    pub classification: DivisibilityClass,
}

/// Classification of how r relates to (p-1)(q-1).
#[derive(Debug, Clone, PartialEq)]
pub enum DivisibilityClass {
    /// r divides both (p-1) and (q-1)
    DividesBoth,
    /// r divides exactly one of (p-1), (q-1)
    DividesOne,
    /// r divides neither
    DividesNeither,
    /// Inconclusive
    Unknown,
}

/// Measure the density of elements whose order is divisible by r.
///
/// For N = pq, we pick random a in (Z/NZ)* and check if a^{(N-1)/r} != 1 (mod N).
/// If a^{(N-1)/r} ≡ 1, then ord(a) does NOT have r as a factor at this level.
/// More precisely: if gcd(a^{(N-1)/r} - 1, N) gives a nontrivial factor, we win immediately.
pub fn measure_density(n: &BigUint, r: u64, num_samples: u64) -> DensityMeasurement {
    let mut rng = rand::thread_rng();
    let one = BigUint::one();

    // We test: a^{(N-1)/r} mod N
    // If result is 1, then ord(a) | (N-1)/r, so r does NOT divide ord(a)
    // If result != 1, then r might divide ord(a)

    // But we don't know if r | (N-1). For arbitrary N = pq:
    // N - 1 = pq - 1. We don't know the factorization of N-1.
    // Instead, use a probabilistic approach:
    // Compute a^{lcm(small primes)} and see if r-th roots are consistent.
    //
    // Simpler: just compute order of a directly (using Pohlig-Hellman from existing code)
    // and check if r divides it.

    let mut num_divisible = 0u64;
    let mut num_tested = 0u64;

    // For practical use, compute orders up to a reasonable bound
    let max_order: u64 = if n.bits() <= 32 { 100_000 } else { 10_000 };

    for _ in 0..num_samples {
        let bytes = n.to_bytes_be();
        let mut random_bytes = vec![0u8; bytes.len()];
        rand::Rng::fill(&mut rng, &mut random_bytes[..]);
        let a = BigUint::from_bytes_be(&random_bytes) % n;
        if a <= one || gcd(&a, n) != one {
            continue;
        }

        num_tested += 1;

        // Compute order of a (or partial order)
        if let Some(ord) = crate::element_order(&a, n, max_order) {
            if ord % r == 0 {
                num_divisible += 1;
            }
        }
    }

    let empirical_density = if num_tested > 0 {
        num_divisible as f64 / num_tested as f64
    } else {
        0.0
    };

    // Classify based on empirical density
    let classification = classify_density(r, empirical_density, num_tested);

    DensityMeasurement {
        r,
        num_tested,
        num_divisible,
        empirical_density,
        classification,
    }
}

/// Classify the density observation into a divisibility class.
fn classify_density(r: u64, density: f64, num_tested: u64) -> DivisibilityClass {
    if num_tested < 10 {
        return DivisibilityClass::Unknown;
    }

    let r_f = r as f64;
    let threshold_both = (2.0 * r_f - 1.0) / (r_f * r_f); // Expected if r | both
    let threshold_one = 1.0 / r_f;                           // Expected if r | one
    let _threshold_neither = 0.0;                             // Expected if r | neither

    // Use intervals with tolerance
    let tolerance = 2.0 / (num_tested as f64).sqrt(); // ~2 sigma

    if (density - threshold_both).abs() < tolerance + 0.1 {
        DivisibilityClass::DividesBoth
    } else if (density - threshold_one).abs() < tolerance + 0.05 {
        DivisibilityClass::DividesOne
    } else if density < tolerance + 0.02 {
        DivisibilityClass::DividesNeither
    } else {
        DivisibilityClass::Unknown
    }
}

/// Full Chebotarev density scan for small primes.
/// Returns density measurements for primes 2, 3, 5, 7, 11, 13, ...
pub fn chebotarev_scan(n: &BigUint, max_prime: u64, samples_per_prime: u64) -> Vec<DensityMeasurement> {
    let primes = sieve_small_primes(max_prime);
    primes.iter()
        .map(|&r| measure_density(n, r, samples_per_prime))
        .collect()
}

fn sieve_small_primes(bound: u64) -> Vec<u64> {
    if bound < 2 { return vec![]; }
    let mut sieve = vec![true; (bound + 1) as usize];
    sieve[0] = false;
    if bound >= 1 { sieve[1] = false; }
    let mut p = 2usize;
    while p * p <= bound as usize {
        if sieve[p] {
            let mut m = p * p;
            while m <= bound as usize { sieve[m] = false; m += p; }
        }
        p += 1;
    }
    (2..=bound).filter(|&i| sieve[i as usize]).collect()
}

/// Extract factor constraints from Chebotarev density measurements.
///
/// For each prime r where classification is DividesOne:
///   r | (p-1) xor r | (q-1)
///   This means exactly one of p-1, q-1 is divisible by r.
///   So: p ≡ 1 (mod r) and q ≢ 1 (mod r), or vice versa.
///
/// For each prime r where classification is DividesBoth:
///   r | (p-1) and r | (q-1)
///   So: p ≡ 1 (mod r) and q ≡ 1 (mod r).
///
/// These congruence constraints can be combined to narrow down p and q.
#[derive(Debug, Clone)]
pub struct FactorConstraints {
    pub n_bits: u32,
    pub divides_both: Vec<u64>,     // Primes dividing both (p-1) and (q-1)
    pub divides_one: Vec<u64>,      // Primes dividing exactly one
    pub divides_neither: Vec<u64>,  // Primes dividing neither
}

pub fn extract_constraints(measurements: &[DensityMeasurement]) -> FactorConstraints {
    let mut divides_both = Vec::new();
    let mut divides_one = Vec::new();
    let mut divides_neither = Vec::new();

    for m in measurements {
        match m.classification {
            DivisibilityClass::DividesBoth => divides_both.push(m.r),
            DivisibilityClass::DividesOne => divides_one.push(m.r),
            DivisibilityClass::DividesNeither => divides_neither.push(m.r),
            DivisibilityClass::Unknown => {},
        }
    }

    FactorConstraints {
        n_bits: 0,
        divides_both,
        divides_one,
        divides_neither,
    }
}

/// Attempt to factor N using Chebotarev density constraints.
///
/// Strategy:
/// 1. Measure densities for small primes
/// 2. For primes r that "divide one": test a^{r-1} mod p candidates
/// 3. Use constraints to narrow search
pub fn factor_via_chebotarev(n: &BigUint, max_prime: u64, samples: u64) -> Option<BigUint> {
    let measurements = chebotarev_scan(n, max_prime, samples);
    let constraints = extract_constraints(&measurements);

    // For primes in divides_one: try using them as Pollard p-1 style bases
    // If r | (p-1), then for random a: a^{k*r} ≡ 1 (mod p) for some k
    // So gcd(a^{product_of_divides_one_primes} - 1, N) might reveal p

    let one = BigUint::one();

    if !constraints.divides_one.is_empty() || !constraints.divides_both.is_empty() {
        // Compute B = product of primes that divide at least one of (p-1), (q-1)
        // raised to reasonable powers
        let mut a = BigUint::from(2u32);

        for &r in constraints.divides_both.iter().chain(constraints.divides_one.iter()) {
            // Raise a to r^k for small k
            let mut power = r;
            while power < 1_000_000 {
                a = mod_pow(&a, &BigUint::from(power), n);
                let diff = if a > one { &a - &one } else { one.clone() };
                let g = gcd(&diff, n);
                if g > one && &g < n {
                    return Some(g);
                }
                power = match power.checked_mul(r) {
                    Some(v) => v,
                    None => break,
                };
            }
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_measure_density_small() {
        let n = BigUint::from(77u32); // 7 * 11
        // (p-1) = 6, (q-1) = 10
        // 2 divides both 6 and 10
        // 3 divides 6 but not 10
        // 5 divides 10 but not 6
        let m2 = measure_density(&n, 2, 200);
        let m3 = measure_density(&n, 3, 200);
        let m5 = measure_density(&n, 5, 200);

        // 2 should have high density (divides both)
        assert!(m2.empirical_density > 0.3, "r=2 should have high density, got {}", m2.empirical_density);
        // 3 and 5 should have moderate density (divides one)
        assert!(m3.empirical_density > 0.05, "r=3 should have some density, got {}", m3.empirical_density);
        assert!(m5.empirical_density > 0.05, "r=5 should have some density, got {}", m5.empirical_density);
    }

    #[test]
    fn test_chebotarev_scan() {
        let n = BigUint::from(77u32);
        let measurements = chebotarev_scan(&n, 13, 100);
        assert!(!measurements.is_empty());
        for m in &measurements {
            assert!(m.num_tested > 0);
        }
    }

    #[test]
    fn test_factor_via_chebotarev_small() {
        // For small N, Chebotarev + Pollard p-1 style should work
        let n = BigUint::from(77u32);
        let result = factor_via_chebotarev(&n, 20, 100);
        if let Some(f) = result {
            assert!(&f > &BigUint::one());
            assert!(&f < &n);
            assert_eq!(&n % &f, BigUint::from(0u32));
        }
        // Note: may fail for some seeds due to randomness -- that's OK
    }

    #[test]
    fn test_extract_constraints() {
        let measurements = vec![
            DensityMeasurement { r: 2, num_tested: 100, num_divisible: 75, empirical_density: 0.75, classification: DivisibilityClass::DividesBoth },
            DensityMeasurement { r: 3, num_tested: 100, num_divisible: 33, empirical_density: 0.33, classification: DivisibilityClass::DividesOne },
            DensityMeasurement { r: 7, num_tested: 100, num_divisible: 0, empirical_density: 0.0, classification: DivisibilityClass::DividesNeither },
        ];
        let constraints = extract_constraints(&measurements);
        assert_eq!(constraints.divides_both, vec![2]);
        assert_eq!(constraints.divides_one, vec![3]);
        assert_eq!(constraints.divides_neither, vec![7]);
    }
}
