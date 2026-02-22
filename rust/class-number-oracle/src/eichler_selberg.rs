//! Eichler-Selberg trace formula for weight 2 modular forms.
//!
//! Computes tr(T_l) on S_2(Gamma_0(N)) as a sum with O(sqrt(l) * d(N)) terms.
//! Each term requires the Hurwitz class number H(D) for D = t^2 - 4l.
//!
//! The key observation: the number of summands is polynomial in log(N) when
//! l is bounded by a polynomial in log(N), but each H(D) computation is
//! currently O(|D|^{1/2}) -- the bottleneck for the spectral factoring approach.

use std::time::Instant;

use crate::class_number::{hurwitz_class_number, kronecker_symbol};

/// Result of computing the Eichler-Selberg trace formula.
#[derive(Debug, Clone)]
pub struct EichlerSelbergResult {
    /// The level N of Gamma_0(N).
    pub n: u64,
    /// The Hecke operator index l.
    pub l: u64,
    /// The computed trace tr(T_l).
    pub trace: f64,
    /// Number of terms in the main sum (= number of valid t values).
    pub num_terms: usize,
    /// The discriminants D = t^2 - 4l arising in the formula.
    pub discriminants: Vec<i64>,
    /// Pairs (D, H(D)) for each discriminant.
    pub class_numbers: Vec<(i64, f64)>,
    /// Total computation time in microseconds.
    pub time_us: u64,
}

/// Collect all discriminants D = t^2 - 4l for |t| < 2*sqrt(l).
///
/// These are the discriminants arising in the Eichler-Selberg trace formula.
/// For the trace formula, we need D < 0 (i.e., t^2 < 4l), so |t| < 2*sqrt(l).
pub fn collect_discriminants(l: u64) -> Vec<i64> {
    let bound = (2.0 * (l as f64).sqrt()).floor() as i64;
    let mut discriminants = Vec::new();

    for t in (-bound)..=bound {
        let d = (t as i64) * (t as i64) - 4 * (l as i64);
        if d < 0 {
            // Must be a valid discriminant: D === 0 or 1 (mod 4)
            let d_mod_4 = ((d % 4) + 4) % 4;
            if d_mod_4 == 0 || d_mod_4 == 1 {
                discriminants.push(d);
            }
        }
    }

    discriminants.sort();
    discriminants.dedup();
    discriminants
}

/// Compute the conductor term C(t, N) in the Eichler-Selberg formula.
///
/// For weight k=2 and level N, the conductor term involves:
///   C(t, l, N) = sum_{s: s^2 | (t^2 - 4l), (t^2-4l)/s^2 fundamental}
///                mu(s) * chi(s) * ...
///
/// For the simplified version with N squarefree and gcd(l, N) = 1:
///   C(t, l, N) = sum_{c | gcd(t^2-4l, N)} c * prod_{p | N/c} (1 - (D_0|p))
///
/// where D_0 is the fundamental discriminant underlying t^2-4l.
///
/// This is a simplification; the full formula is more involved.
pub fn conductor_sum(t: i64, l: u64, n: u64) -> f64 {
    let d = t * t - 4 * (l as i64);
    if d >= 0 {
        return 0.0;
    }

    // Find the fundamental discriminant D_0 and conductor f such that D = D_0 * f^2
    let (d0, _f) = fundamental_discriminant(d);

    // For squarefree N with gcd(l, N) = 1:
    // The contribution depends on how D_0 splits in the prime factors of N.
    //
    // Simplified formula for the conductor term:
    // For each prime p | N:
    //   if D_0 mod p == 0 (ramified): factor = 1
    //   if (D_0|p) = 1 (split): factor = 2
    //   if (D_0|p) = -1 (inert): factor = 0
    //
    // The full term is the product over primes dividing N.
    // But this is an approximation; the actual formula involves
    // a sum over orders in Q(sqrt(D)).

    let primes = prime_factors(n);
    let mut product = 1.0f64;

    for &p in &primes {
        let kp = kronecker_symbol(d0, p);
        // For weight 2, the local factor at p for the trace is:
        // (1 + 1/p) if (D_0|p) = 0 (ramified)
        // (1 - (D_0|p)/p) ... this needs careful treatment
        //
        // Simplified: the multiplicity of embeddings of the order Z[(D+sqrt(D))/2]
        // into M_2(Z_p) contributes to the count.
        // For p not dividing D: if (D|p) = 1, contributes 2; if -1, contributes 0.
        // For p dividing D: contributes 1.
        match kp {
            0 => {
                // Ramified: p divides D_0
                product *= 1.0;
            }
            1 => {
                // Split: D_0 is a QR mod p
                product *= 2.0;
            }
            -1 => {
                // Inert: D_0 is a QNR mod p
                product *= 0.0;
            }
            _ => {}
        }
    }

    product
}

/// Extract the fundamental discriminant D_0 and conductor f from D = D_0 * f^2.
fn fundamental_discriminant(d: i64) -> (i64, u64) {
    let mut abs_d = (-d) as u64;
    let sign = -1i64;
    let mut f = 1u64;

    // Remove square factors from abs_d
    let mut p = 2u64;
    while p * p <= abs_d {
        while abs_d % (p * p) == 0 {
            abs_d /= p * p;
            f *= p;
        }
        p += 1;
    }

    let d0 = sign * (abs_d as i64);

    // Ensure D_0 is congruent to 0 or 1 mod 4
    let d0_mod_4 = ((d0 % 4) + 4) % 4;
    if d0_mod_4 == 0 || d0_mod_4 == 1 {
        (d0, f)
    } else {
        // D_0 === 2 or 3 mod 4, need to absorb a factor of 4
        // Actually if d0 === 2 or 3 mod 4, then the fundamental disc is 4*d0
        // and f should be divided by 2
        // This happens when the squarefree part is 2 or 3 mod 4
        if f % 2 == 0 {
            (4 * d0, f / 2)
        } else {
            // d0 is already squarefree but not 0,1 mod 4
            // The fundamental discriminant for Q(sqrt(d0)) is 4*d0
            (4 * d0, f)
        }
    }
}

/// Compute prime factorization and return list of prime factors (without multiplicity).
fn prime_factors(n: u64) -> Vec<u64> {
    let mut factors = Vec::new();
    let mut temp = n;
    let mut p = 2u64;

    while p * p <= temp {
        if temp % p == 0 {
            factors.push(p);
            while temp % p == 0 {
                temp /= p;
            }
        }
        p += 1;
    }
    if temp > 1 {
        factors.push(temp);
    }

    factors
}

/// Number of divisors of n.
fn num_divisors(n: u64) -> u64 {
    let mut count = 1u64;
    let mut temp = n;
    let mut p = 2u64;

    while p * p <= temp {
        if temp % p == 0 {
            let mut exp = 0u64;
            while temp % p == 0 {
                temp /= p;
                exp += 1;
            }
            count *= exp + 1;
        }
        p += 1;
    }
    if temp > 1 {
        count *= 2;
    }

    count
}

/// Compute tr(T_l) on S_2(Gamma_0(N)) via the Eichler-Selberg trace formula.
///
/// For weight k=2, the formula is:
///   tr(T_l) = -1/2 * sum_{|t| < 2*sqrt(l)} H(t^2 - 4l) * C(t, l, N)
///             - 1/2 * sum_{d | l} min(d, l/d)   (identity contribution)
///             + (genus - 1) * sigma_1(l)          (... correction for old forms)
///
/// The main sum has O(sqrt(l)) terms, each requiring H(D) for D = t^2 - 4l.
/// The key question: can we compute these H(D) values faster than O(|D|^{1/2}) each?
pub fn eichler_selberg_trace(n: u64, l: u64) -> EichlerSelbergResult {
    let start = Instant::now();

    assert!(l > 0, "l must be positive");
    assert!(n > 0, "N must be positive");

    let bound = (2.0 * (l as f64).sqrt()).floor() as i64;

    let mut discriminants = Vec::new();
    let mut class_numbers = Vec::new();
    let mut main_sum = 0.0f64;

    // Main sum: over t with |t| <= 2*sqrt(l), t^2 - 4l < 0
    for t in (-bound)..=bound {
        let d = (t as i64) * (t as i64) - 4 * (l as i64);
        if d >= 0 {
            continue;
        }

        // Check valid discriminant
        let d_mod_4 = ((d % 4) + 4) % 4;
        if d_mod_4 != 0 && d_mod_4 != 1 {
            continue;
        }

        let h_d = hurwitz_class_number(d);
        let c_term = conductor_sum(t, l, n);

        discriminants.push(d);
        class_numbers.push((d, h_d));

        main_sum += h_d * c_term;
    }

    // Main contribution
    let mut trace = -0.5 * main_sum;

    // Identity contribution: -1/2 * sum_{d | l} min(d, l/d)
    // This accounts for the identity Hecke operator
    let mut identity_sum = 0.0f64;
    for d in 1..=l {
        if l % d == 0 {
            let other = l / d;
            identity_sum += d.min(other) as f64;
        }
    }
    trace -= 0.5 * identity_sum;

    // Note: for a complete implementation, additional correction terms are needed
    // for old forms, cusps, etc. This is the core formula.

    let time_us = start.elapsed().as_micros() as u64;

    // Deduplicate discriminants for reporting
    let mut unique_discs = discriminants.clone();
    unique_discs.sort();
    unique_discs.dedup();

    let mut unique_class_nums: Vec<(i64, f64)> = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for &(d, h) in &class_numbers {
        if seen.insert(d) {
            unique_class_nums.push((d, h));
        }
    }

    EichlerSelbergResult {
        n,
        l,
        trace,
        num_terms: discriminants.len(),
        discriminants: unique_discs,
        class_numbers: unique_class_nums,
        time_us,
    }
}

/// Analyze the computational cost of the Eichler-Selberg formula.
///
/// Returns estimates of:
/// - Number of terms in the main sum: O(sqrt(l))
/// - Total cost with naive H(D): O(sqrt(l) * sqrt(l)) = O(l)
/// - Number of divisors of N (affects conductor sum): d(N)
/// - Theoretical advantage if H(D) were subexponential in |D|
#[derive(Debug, Clone)]
pub struct TraceFormulaComplexity {
    pub n: u64,
    pub l: u64,
    pub num_main_terms: usize,
    pub num_divisors_n: u64,
    pub max_abs_d: u64,
    pub avg_abs_d: f64,
    pub naive_total_ops_estimate: f64,
    pub shanks_total_ops_estimate: f64,
}

/// Estimate the computational complexity of the trace formula for given N and l.
pub fn analyze_complexity(n: u64, l: u64) -> TraceFormulaComplexity {
    let discriminants = collect_discriminants(l);
    let num_main_terms = discriminants.len();
    let d_n = num_divisors(n);

    let max_abs_d = discriminants.iter().map(|&d| (-d) as u64).max().unwrap_or(0);
    let avg_abs_d = if num_main_terms > 0 {
        discriminants.iter().map(|&d| (-d) as f64).sum::<f64>() / num_main_terms as f64
    } else {
        0.0
    };

    // Naive: each H(D) costs O(|D|^{1/2}), there are O(sqrt(l)) terms
    let naive_total: f64 = discriminants
        .iter()
        .map(|&d| ((-d) as f64).sqrt())
        .sum();

    // Shanks: each H(D) costs O(|D|^{1/4}), there are O(sqrt(l)) terms
    let shanks_total: f64 = discriminants
        .iter()
        .map(|&d| ((-d) as f64).powf(0.25))
        .sum();

    TraceFormulaComplexity {
        n,
        l,
        num_main_terms,
        num_divisors_n: d_n,
        max_abs_d,
        avg_abs_d,
        naive_total_ops_estimate: naive_total * d_n as f64,
        shanks_total_ops_estimate: shanks_total * d_n as f64,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_collect_discriminants() {
        // For l=2: t^2 - 8, |t| < 2*sqrt(2) ~ 2.83, so t in {-2, -1, 0, 1, 2}
        // t=0: D = -8 (mod 4 = 0, valid)
        // t=1: D = 1-8 = -7 (mod 4 = 1, valid)
        // t=-1: D = 1-8 = -7 (same)
        // t=2: D = 4-8 = -4 (mod 4 = 0, valid)
        // t=-2: D = 4-8 = -4 (same)
        let discs = collect_discriminants(2);
        assert!(discs.contains(&-8), "Should contain D=-8");
        assert!(discs.contains(&-7), "Should contain D=-7");
        assert!(discs.contains(&-4), "Should contain D=-4");
    }

    #[test]
    fn test_collect_discriminants_l5() {
        // For l=5: |t| < 2*sqrt(5) ~ 4.47, so t in {-4,-3,-2,-1,0,1,2,3,4}
        let discs = collect_discriminants(5);
        // t=0: D = -20 (mod4=0 valid)
        // t=1: D = 1-20 = -19 (mod4=1 valid)
        // t=2: D = 4-20 = -16 (mod4=0 valid)
        // t=3: D = 9-20 = -11 (mod4=1 valid)
        // t=4: D = 16-20 = -4 (mod4=0 valid)
        assert!(discs.contains(&-20));
        assert!(discs.contains(&-19));
        assert!(discs.contains(&-16));
        assert!(discs.contains(&-11));
        assert!(discs.contains(&-4));
    }

    #[test]
    fn test_fundamental_discriminant() {
        // -4 is fundamental (not -1 * 4 since -4 itself is fundamental)
        let (d0, f) = fundamental_discriminant(-4);
        assert_eq!(d0, -4);
        assert_eq!(f, 1);

        // -8 = -8 * 1^2, fundamental
        let (d0, f) = fundamental_discriminant(-8);
        assert_eq!(d0, -8);
        assert_eq!(f, 1);

        // -12 = -3 * 4 = -3 * 2^2
        let (d0, f) = fundamental_discriminant(-12);
        assert_eq!(d0, -3);
        assert_eq!(f, 2);

        // -16 = -4 * 4 = -4 * 2^2
        let (d0, f) = fundamental_discriminant(-16);
        assert_eq!(d0, -4);
        assert_eq!(f, 2);
    }

    #[test]
    fn test_eichler_selberg_trace_runs() {
        // Basic smoke test: formula runs without panic
        let result = eichler_selberg_trace(11, 2);
        assert_eq!(result.n, 11);
        assert_eq!(result.l, 2);
        assert!(result.num_terms > 0);
        assert!(!result.discriminants.is_empty());
    }

    #[test]
    fn test_eichler_selberg_trace_n77() {
        // N=77=7*11 is a semiprime
        let result = eichler_selberg_trace(77, 2);
        assert_eq!(result.n, 77);
        assert!(result.num_terms > 0);
        // The trace should be a finite number
        assert!(result.trace.is_finite());
    }

    #[test]
    fn test_complexity_analysis() {
        let complexity = analyze_complexity(77, 2);
        assert!(complexity.num_main_terms > 0);
        assert_eq!(complexity.num_divisors_n, 4); // 77 = 7 * 11, divisors: 1, 7, 11, 77
        assert!(complexity.max_abs_d > 0);
    }

    #[test]
    fn test_prime_factors() {
        assert_eq!(prime_factors(77), vec![7, 11]);
        assert_eq!(prime_factors(12), vec![2, 3]);
        assert_eq!(prime_factors(1), Vec::<u64>::new());
        assert_eq!(prime_factors(7), vec![7]);
    }
}
