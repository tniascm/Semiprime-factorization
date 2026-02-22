//! Quantum-inspired classical factoring algorithms.
//!
//! Explores classical simulations of quantum period-finding and
//! dequantization approaches inspired by Ewin Tang's work.

use factoring_core::gcd;
use num_bigint::BigUint;
use num_traits::{One, Zero};
use rand::Rng;
use std::collections::HashMap;

/// Classical period-finding for f(x) = a^x mod N.
/// This is the core subroutine that Shor's algorithm accelerates with QFT.
/// Classical version is exponential but useful for understanding the structure.
pub fn classical_period_find(a: &BigUint, n: &BigUint, max_period: u64) -> Option<u64> {
    let one = BigUint::one();
    let mut current = a.clone();

    for period in 1..=max_period {
        if current == one {
            return Some(period);
        }
        current = (current * a) % n;
    }

    None
}

/// Attempt factoring via classical period-finding (Shor's approach without quantum).
/// For small numbers only — exponential in general.
pub fn period_factor(n: &BigUint, max_attempts: u32) -> Option<BigUint> {
    let one = BigUint::one();
    let _two = BigUint::from(2u32);

    for attempt in 2..max_attempts {
        let a = BigUint::from(attempt);
        let g = gcd(&a, n);
        if g > one && g < *n {
            return Some(g);
        }

        // Find period of a^x mod N
        let max_period = 10_000u64; // Small bound for classical simulation
        if let Some(period) = classical_period_find(&a, n, max_period) {
            if period % 2 == 0 {
                let half_period = BigUint::from(period / 2);
                let a_half = a.modpow(&half_period, n);

                let candidate1 = if a_half > one {
                    gcd(&(&a_half - &one), n)
                } else {
                    one.clone()
                };
                let candidate2 = gcd(&(&a_half + &one), n);

                if candidate1 > one && candidate1 < *n {
                    return Some(candidate1);
                }
                if candidate2 > one && candidate2 < *n {
                    return Some(candidate2);
                }
            }
        }
    }

    None
}

/// Regev-inspired multidimensional approach (classical simulation).
/// Uses multiple bases simultaneously — inspired by Regev's use of
/// high-dimensional lattice geometry for quantum factoring.
pub fn multidim_period_search(n: &BigUint, bases: &[BigUint], max_exp: u64) -> Option<BigUint> {
    let one = BigUint::one();

    if bases.is_empty() {
        return None;
    }

    // 1. For each base, find the individual period of a_i^x mod n
    let mut periods: Vec<u64> = Vec::new();
    for base in bases {
        // Skip bases that share a common factor with n (trivial factoring)
        let g = gcd(base, n);
        if g > one && g < *n {
            return Some(g);
        }

        if let Some(period) = classical_period_find(base, n, max_exp) {
            periods.push(period);
        }
        // If we can't find a period for this base, just skip it
    }

    if periods.is_empty() {
        return None;
    }

    // 2. Compute the LCM of all individual periods to find a combined period
    //    that works simultaneously for all bases
    let mut combined_period = periods[0];
    for &p in &periods[1..] {
        combined_period = lcm(combined_period, p);
        // Guard against overflow / excessively large periods
        if combined_period > max_exp * max_exp {
            break;
        }
    }

    // 3. If the combined period is even, try the Shor-style extraction
    //    for each base: gcd(a^(r/2) - 1, n) or gcd(a^(r/2) + 1, n)
    if combined_period % 2 == 0 {
        let half_period = BigUint::from(combined_period / 2);

        for base in bases {
            let a_half = base.modpow(&half_period, n);

            if a_half > one {
                let candidate = gcd(&(&a_half - &one), n);
                if candidate > one && candidate < *n {
                    return Some(candidate);
                }
            }

            let candidate = gcd(&(&a_half + &one), n);
            if candidate > one && candidate < *n {
                return Some(candidate);
            }
        }
    }

    // 4. Also try with each individual period (in case LCM was too large
    //    or odd, but an individual period is even)
    for (i, &period) in periods.iter().enumerate() {
        if period % 2 != 0 {
            continue;
        }
        let half_period = BigUint::from(period / 2);
        let base = &bases[i];
        let a_half = base.modpow(&half_period, n);

        if a_half > one {
            let candidate = gcd(&(&a_half - &one), n);
            if candidate > one && candidate < *n {
                return Some(candidate);
            }
        }

        let candidate = gcd(&(&a_half + &one), n);
        if candidate > one && candidate < *n {
            return Some(candidate);
        }
    }

    None
}

/// Compute the least common multiple of two u64 values.
fn lcm(a: u64, b: u64) -> u64 {
    if a == 0 || b == 0 {
        return 0;
    }
    let g = gcd_u64(a, b);
    a / g * b
}

/// GCD for u64 values (Euclidean algorithm).
fn gcd_u64(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

// ─── Tang-inspired sample-and-query dequantization ───────────────────────────

/// Oracle that supports importance sampling of matrix rows and entry queries,
/// following the dequantization framework introduced by Ewin Tang.
///
/// Quantum machine-learning algorithms achieve speedups partly because they can
/// prepare amplitude-encoded states efficiently. Tang showed that if classical
/// algorithms have sampling-and-query access to the input data (i.e., they can
/// sample row indices proportional to squared norms and query individual entries),
/// many of those speedups can be matched classically, up to polynomial factors.
///
/// This struct precomputes the squared row norms so that `sample_row` runs in
/// O(rows) time (linear scan with cumulative distribution).
pub struct SampleQueryOracle {
    matrix: Vec<Vec<f64>>,
    row_norms_sq: Vec<f64>,
    total_norm_sq: f64,
}

impl SampleQueryOracle {
    /// Build the oracle, precomputing squared row norms and the total Frobenius
    /// norm squared.
    pub fn new(matrix: Vec<Vec<f64>>) -> Self {
        let row_norms_sq: Vec<f64> = matrix
            .iter()
            .map(|row| row.iter().map(|v| v * v).sum())
            .collect();
        let total_norm_sq: f64 = row_norms_sq.iter().sum();
        Self {
            matrix,
            row_norms_sq,
            total_norm_sq,
        }
    }

    /// Sample a row index proportional to its squared Euclidean norm
    /// (importance sampling). This mirrors the quantum ability to prepare a
    /// state whose amplitudes encode the matrix row norms.
    pub fn sample_row(&self, rng: &mut impl Rng) -> usize {
        let threshold: f64 = rng.gen::<f64>() * self.total_norm_sq;
        let mut cumulative = 0.0;
        for (i, &norm_sq) in self.row_norms_sq.iter().enumerate() {
            cumulative += norm_sq;
            if cumulative >= threshold {
                return i;
            }
        }
        // Fallback — should only be reached due to floating-point rounding.
        self.row_norms_sq.len() - 1
    }

    /// Query a single matrix entry.
    pub fn query_entry(&self, i: usize, j: usize) -> f64 {
        self.matrix[i][j]
    }

    /// Produce a low-rank approximation of the stored matrix using Tang's
    /// random-projection technique.
    ///
    /// 1. Sample `num_samples` row indices (with replacement) proportional to
    ///    squared row norms.
    /// 2. Keep the first `rank` *distinct* sampled rows (or as many as we get).
    /// 3. Normalize each selected row to unit length to form an approximate
    ///    orthonormal basis.
    /// 4. Project the full matrix onto that basis, yielding a rank-`rank`
    ///    approximation.
    ///
    /// The returned matrix has the same dimensions as the original.
    pub fn low_rank_approx(
        &self,
        rank: usize,
        num_samples: usize,
        rng: &mut impl Rng,
    ) -> Vec<Vec<f64>> {
        let rows = self.matrix.len();
        if rows == 0 {
            return vec![];
        }
        let cols = self.matrix[0].len();

        // Step 1 & 2: sample distinct row indices.
        let mut seen = Vec::<usize>::new();
        for _ in 0..num_samples {
            if seen.len() >= rank {
                break;
            }
            let idx = self.sample_row(rng);
            if !seen.contains(&idx) {
                seen.push(idx);
            }
        }

        // Step 3: build a (possibly non-orthogonal) basis from the sampled rows
        // by normalizing each row.
        let basis: Vec<Vec<f64>> = seen
            .iter()
            .map(|&idx| {
                let row = &self.matrix[idx];
                let norm: f64 = row.iter().map(|v| v * v).sum::<f64>().sqrt();
                if norm > 1e-15 {
                    row.iter().map(|v| v / norm).collect()
                } else {
                    vec![0.0; cols]
                }
            })
            .collect();

        // Step 4: project every row of the original matrix onto the basis.
        //   approx_row = sum_k ( row · basis_k ) * basis_k
        let mut result = vec![vec![0.0; cols]; rows];
        for (r, original_row) in self.matrix.iter().enumerate() {
            for basis_vec in &basis {
                let dot: f64 = original_row
                    .iter()
                    .zip(basis_vec.iter())
                    .map(|(a, b)| a * b)
                    .sum();
                for (c, bv) in basis_vec.iter().enumerate() {
                    result[r][c] += dot * bv;
                }
            }
        }

        result
    }
}

// ─── Quantum walk on Cayley graph ────────────────────────────────────────────

/// Simulate a classical random walk on the Cayley graph of the multiplicative
/// group (Z/nZ)*.
///
/// The Cayley graph has as vertices the elements of (Z/nZ)* and as edges the
/// connections v -> v*g (mod n) for each generator g. Quantum walks on such
/// graphs can reveal hidden subgroup structure exponentially faster than
/// classical walks; this function provides the classical baseline.
///
/// Returns the top visited nodes together with their visit counts, sorted by
/// descending frequency.
pub fn cayley_graph_walk(
    n: &BigUint,
    generators: &[BigUint],
    steps: usize,
) -> Vec<(BigUint, u64)> {
    if generators.is_empty() || n.is_zero() {
        return vec![];
    }

    let mut rng = rand::thread_rng();
    let one = BigUint::one();
    let mut current = one.clone(); // start at the identity
    let mut visits: HashMap<BigUint, u64> = HashMap::new();
    *visits.entry(current.clone()).or_insert(0) += 1;

    let gen_count = generators.len();

    for _ in 0..steps {
        let idx = rng.gen_range(0..gen_count);
        current = (&current * &generators[idx]) % n;
        *visits.entry(current.clone()).or_insert(0) += 1;
    }

    let mut result: Vec<(BigUint, u64)> = visits.into_iter().collect();
    result.sort_by(|a, b| b.1.cmp(&a.1));
    result
}

// ─── Grover-style brute force estimation ─────────────────────────────────────

/// Quantifies the quadratic speedup Grover's algorithm would provide for a
/// brute-force search over `2^search_space_bits` elements with the given
/// success probability per evaluation.
#[derive(Debug, Clone)]
pub struct GroverEstimate {
    pub classical_evals: f64,
    pub quantum_evals: f64,
    pub speedup_factor: f64,
}

/// Compute Grover speedup estimates.
///
/// * Classical: expected evaluations = 1 / p
/// * Quantum (Grover): expected evaluations = (π / 4) / √p
/// * Speedup factor = classical / quantum
pub fn grover_speedup_estimate(
    _search_space_bits: u32,
    success_probability: f64,
) -> GroverEstimate {
    assert!(
        success_probability > 0.0 && success_probability <= 1.0,
        "success_probability must be in (0, 1]"
    );

    let classical_evals = 1.0 / success_probability;
    let quantum_evals = std::f64::consts::FRAC_PI_4 / success_probability.sqrt();
    let speedup_factor = classical_evals / quantum_evals;

    GroverEstimate {
        classical_evals,
        quantum_evals,
        speedup_factor,
    }
}

// ─── Continued fraction expansion ────────────────────────────────────────────

/// Compute the continued fraction expansion of numerator / denominator,
/// returning at most `max_terms` partial quotients.
///
/// The continued fraction representation [a0; a1, a2, ...] satisfies
///   numerator/denominator = a0 + 1/(a1 + 1/(a2 + ...))
pub fn continued_fraction(
    mut numerator: u64,
    mut denominator: u64,
    max_terms: usize,
) -> Vec<u64> {
    let mut terms = Vec::new();
    while denominator != 0 && terms.len() < max_terms {
        let quotient = numerator / denominator;
        terms.push(quotient);
        let remainder = numerator % denominator;
        numerator = denominator;
        denominator = remainder;
    }
    terms
}

/// Compute the convergents (rational approximations p_k/q_k) from a continued
/// fraction expansion.
///
/// Returns a vector of (numerator, denominator) pairs.  Each convergent is the
/// best rational approximation with denominator ≤ q_k.
pub fn convergents(cf: &[u64]) -> Vec<(u64, u64)> {
    if cf.is_empty() {
        return vec![];
    }

    let mut result = Vec::with_capacity(cf.len());

    // p_{-1} = 1, p_{-2} = 0  (standard recurrence seed)
    // q_{-1} = 0, q_{-2} = 1
    let mut p_prev2: u64 = 0;
    let mut p_prev1: u64 = 1;
    let mut q_prev2: u64 = 1;
    let mut q_prev1: u64 = 0;

    for &a in cf {
        let p = a.checked_mul(p_prev1).and_then(|v| v.checked_add(p_prev2));
        let q = a.checked_mul(q_prev1).and_then(|v| v.checked_add(q_prev2));
        match (p, q) {
            (Some(pn), Some(qn)) => {
                result.push((pn, qn));
                p_prev2 = p_prev1;
                p_prev1 = pn;
                q_prev2 = q_prev1;
                q_prev1 = qn;
            }
            _ => break, // overflow — stop early
        }
    }

    result
}

/// Given a simulated quantum phase-estimation measurement, use continued
/// fractions to extract the period and attempt Shor-style factoring.
///
/// In Shor's algorithm the quantum register yields a value `measurement`
/// close to `j * modulus / r` for some integer `j`, where `r` is the unknown
/// period.  We compute the continued fraction of `measurement / modulus` and
/// check each convergent denominator as a candidate period.
///
/// `n` is the number being factored.  Returns `Some(factor)` on success.
pub fn period_from_measurement(
    measurement: u64,
    modulus: u64,
    n: &BigUint,
) -> Option<BigUint> {
    let one = BigUint::one();
    let cf = continued_fraction(measurement, modulus, 64);
    let convs = convergents(&cf);

    for &(_p, q) in &convs {
        if q == 0 || q == 1 {
            continue;
        }

        let period = q;

        // Verify the period: check that a^period ≡ 1 (mod n) for small bases.
        // We try a = 2 as a default base.
        let a = BigUint::from(2u32);
        let g = gcd(&a, n);
        if g > one && g < *n {
            return Some(g); // lucky: gcd already gave a factor
        }

        let result = a.modpow(&BigUint::from(period), n);
        if result != one {
            continue; // not a valid period
        }

        // Valid period found — try Shor's extraction
        if period % 2 != 0 {
            continue;
        }

        let half_period = BigUint::from(period / 2);
        let a_half = a.modpow(&half_period, n);

        if a_half > one {
            let candidate = gcd(&(&a_half - &one), n);
            if candidate > one && candidate < *n {
                return Some(candidate);
            }
        }

        let candidate = gcd(&(&a_half + &one), n);
        if candidate > one && candidate < *n {
            return Some(candidate);
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_period_find() {
        let n = BigUint::from(15u32);
        let a = BigUint::from(2u32);
        // 2^1=2, 2^2=4, 2^3=8, 2^4=1 (mod 15), period=4
        let period = classical_period_find(&a, &n, 100);
        assert_eq!(period, Some(4));
    }

    #[test]
    fn test_period_factor() {
        let n = BigUint::from(15u32);
        let factor = period_factor(&n, 10);
        assert!(factor.is_some());
        let f = factor.unwrap();
        assert!(f == BigUint::from(3u32) || f == BigUint::from(5u32));
    }

    // ── Sample-Query Oracle tests ────────────────────────────────────────

    #[test]
    fn test_sample_query_oracle() {
        let matrix = vec![
            vec![3.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 2.0],
        ];
        let oracle = SampleQueryOracle::new(matrix);

        // Check precomputed norms: row0=9, row1=1, row2=4 => total=14
        assert!((oracle.total_norm_sq - 14.0).abs() < 1e-10);

        // Verify entry queries
        assert!((oracle.query_entry(0, 0) - 3.0).abs() < 1e-10);
        assert!((oracle.query_entry(1, 1) - 1.0).abs() < 1e-10);
        assert!((oracle.query_entry(2, 2) - 2.0).abs() < 1e-10);
        assert!((oracle.query_entry(0, 1) - 0.0).abs() < 1e-10);

        // Sampling: with enough samples, row 0 (norm_sq=9) should be sampled
        // more often than row 1 (norm_sq=1).
        let mut rng = rand::thread_rng();
        let mut counts = [0u32; 3];
        for _ in 0..10_000 {
            let idx = oracle.sample_row(&mut rng);
            counts[idx] += 1;
        }
        // Row 0 should have roughly 9/14 ≈ 64% of samples
        assert!(counts[0] > counts[1], "Row 0 should be sampled more than row 1");
        assert!(counts[0] > counts[2], "Row 0 should be sampled more than row 2");

        // Low-rank approximation: dimensions should match original
        let approx = oracle.low_rank_approx(2, 100, &mut rng);
        assert_eq!(approx.len(), 3, "Approx should have 3 rows");
        for row in &approx {
            assert_eq!(row.len(), 3, "Each approx row should have 3 columns");
        }
    }

    // ── Cayley graph walk tests ──────────────────────────────────────────

    #[test]
    fn test_cayley_walk() {
        let n = BigUint::from(15u32);
        let generators = vec![BigUint::from(2u32), BigUint::from(4u32)];
        let result = cayley_graph_walk(&n, &generators, 1000);

        assert!(!result.is_empty(), "Walk should visit at least one node");

        // All visited nodes must be coprime to 15.
        // Since we start at 1 and multiply by generators coprime to 15,
        // every visited node must be coprime to 15.
        for (node, count) in &result {
            assert!(*count > 0);
            let g = gcd(node, &n);
            assert_eq!(
                g,
                BigUint::one(),
                "Node {} should be coprime to 15 but gcd = {}",
                node,
                g
            );
        }

        // The identity (1) should be visited at least once (it's the start).
        let one_visits = result.iter().find(|(node, _)| *node == BigUint::one());
        assert!(
            one_visits.is_some(),
            "The identity element should be visited at least once"
        );
    }

    // ── Grover estimate tests ────────────────────────────────────────────

    #[test]
    fn test_grover_estimate() {
        // For p = 1/N, classical = N, quantum ≈ (π/4)√N
        // speedup_factor = N / ((π/4)√N) = (4/π)√N
        let p = 0.01; // 1 in 100
        let est = grover_speedup_estimate(10, p);

        assert!((est.classical_evals - 100.0).abs() < 1e-10);

        // quantum_evals = (π/4) / √0.01 = (π/4) / 0.1 = 10π/4 ≈ 7.854
        let expected_quantum = std::f64::consts::FRAC_PI_4 / 0.1;
        assert!(
            (est.quantum_evals - expected_quantum).abs() < 1e-10,
            "quantum_evals should be π/(4*√p)"
        );

        // speedup_factor = classical / quantum ≈ 100 / 7.854 ≈ 12.73
        // This should be close to (4/π)*√(1/p) = (4/π)*10 ≈ 12.73
        let expected_speedup = (4.0 / std::f64::consts::PI) * (1.0 / p).sqrt();
        assert!(
            (est.speedup_factor - expected_speedup).abs() < 1e-6,
            "speedup_factor should be (4/π)*√(1/p), got {}",
            est.speedup_factor
        );

        // Verify the quadratic relationship: speedup_factor ≈ (4/π)*√(1/p)
        for &prob in &[0.001, 0.01, 0.1, 0.5] {
            let e = grover_speedup_estimate(20, prob);
            let expected = (4.0 / std::f64::consts::PI) * (1.0 / prob).sqrt();
            assert!(
                (e.speedup_factor - expected).abs() < 1e-6,
                "For p={}, expected speedup {}, got {}",
                prob,
                expected,
                e.speedup_factor
            );
        }
    }

    // ── Continued fraction tests ─────────────────────────────────────────

    #[test]
    fn test_continued_fraction() {
        // CF(31/13) = [2; 2, 1, 1, 2]
        // 31/13 = 2 + 5/13 → 13/5 = 2 + 3/5 → 5/3 = 1 + 2/3 → 3/2 = 1 + 1/2 → 2/1 = 2
        let cf = continued_fraction(31, 13, 20);
        assert_eq!(cf, vec![2, 2, 1, 1, 2]);

        // Convergents of [2; 2, 1, 1, 2]:
        //   k=0: p=2,  q=1   → 2/1
        //   k=1: p=5,  q=2   → 5/2
        //   k=2: p=7,  q=3   → 7/3
        //   k=3: p=12, q=5   → 12/5
        //   k=4: p=31, q=13  → 31/13
        let convs = convergents(&cf);
        assert_eq!(
            convs,
            vec![(2, 1), (5, 2), (7, 3), (12, 5), (31, 13)]
        );

        // Verify the last convergent reconstructs the original fraction.
        let (p, q) = convs.last().unwrap();
        assert_eq!(*p, 31);
        assert_eq!(*q, 13);

        // Edge case: CF(7/1) = [7]
        let cf2 = continued_fraction(7, 1, 10);
        assert_eq!(cf2, vec![7]);

        // Edge case: CF(0/5) = [0]
        let cf3 = continued_fraction(0, 5, 10);
        assert_eq!(cf3, vec![0]);
    }

    // ── Period-from-measurement tests ────────────────────────────────────

    #[test]
    fn test_period_from_measurement() {
        // Factor n = 15.  The period of 2^x mod 15 is r = 4.
        // In Shor's algorithm with a modulus (register size) of Q = 256,
        // a valid measurement would be close to j*Q/r = j*64 for j=0..3.
        // measurement = 64 → CF(64/256) = CF(1/4) = [0; 4] → convergent 1/4 → period = 4
        let n = BigUint::from(15u32);
        let factor = period_from_measurement(64, 256, &n);
        assert!(
            factor.is_some(),
            "Should find a factor of 15 from measurement=64, Q=256"
        );
        let f = factor.unwrap();
        assert!(
            f == BigUint::from(3u32) || f == BigUint::from(5u32),
            "Factor should be 3 or 5, got {}",
            f
        );

        // measurement = 128 → CF(128/256) = CF(1/2) = [0; 2] → period = 2
        // 2^1 = 2 mod 15 (not 1), so period 2 is invalid for base 2.
        // But CF also gives trivial convergents, so this might not find anything.
        // That's fine — the function should return None gracefully.
        let factor2 = period_from_measurement(128, 256, &n);
        // period=2: 2^2=4 mod 15 ≠ 1, so this should fail.
        // The function should return None.
        assert!(
            factor2.is_none(),
            "measurement=128 should not yield a valid factor (period 2 invalid for base 2 mod 15)"
        );

        // measurement = 192 → CF(192/256) = CF(3/4) = [0; 1, 3] → convergents: 0/1, 1/1, 3/4
        // Convergent 3/4 → period = 4. 2^4 = 16 ≡ 1 (mod 15), valid!
        let factor3 = period_from_measurement(192, 256, &n);
        assert!(
            factor3.is_some(),
            "Should find a factor of 15 from measurement=192, Q=256"
        );
        let f3 = factor3.unwrap();
        assert!(
            f3 == BigUint::from(3u32) || f3 == BigUint::from(5u32),
            "Factor should be 3 or 5, got {}",
            f3
        );
    }
}
