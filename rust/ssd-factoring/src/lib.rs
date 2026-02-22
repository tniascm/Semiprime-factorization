pub mod binary_lift;
pub mod ntt_domain;
pub mod crt_parallel;

use std::time::Instant;

/// Report comparing sequential vs parallel computation of N mod d for each divisor d.
#[derive(Debug, Clone)]
pub struct DualityReport {
    pub strategy_name: String,
    pub n: u64,
    pub divisors_tested: usize,
    pub sequential_results: Vec<u64>,
    pub parallel_results: Vec<u64>,
    pub results_match: bool,
    pub dimensionality: usize,
    pub sequential_time_ns: u64,
    pub parallel_time_ns: u64,
    pub speedup: f64,
}

impl std::fmt::Display for DualityReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "  Strategy:       {}", self.strategy_name)?;
        writeln!(f, "  N:              {}", self.n)?;
        writeln!(f, "  Divisors:       {}", self.divisors_tested)?;
        writeln!(f, "  Dimensionality: {}", self.dimensionality)?;
        writeln!(f, "  Results match:  {}", self.results_match)?;
        writeln!(f, "  Seq time (ns):  {}", self.sequential_time_ns)?;
        writeln!(f, "  Par time (ns):  {}", self.parallel_time_ns)?;
        writeln!(f, "  Speedup:        {:.4}x", self.speedup)?;
        Ok(())
    }
}

/// Trait for SSD linearization strategies.
///
/// Each strategy provides a sequential baseline (N mod d for each d) and
/// a parallel formulation that attempts to compute the same results via
/// a matrix/transform-based approach inspired by the State Space Duality theorem.
pub trait SsdFormulation {
    /// Human-readable name of this strategy.
    fn name(&self) -> &str;

    /// Sequential baseline: compute N mod d for each divisor d.
    fn sequential(&self, n: u64, divisors: &[u64]) -> Vec<u64> {
        trial_division_sequential(n, divisors)
    }

    /// Parallel/linearized form: compute N mod d for each divisor d
    /// using the strategy's matrix or transform-based approach.
    fn parallel(&self, n: u64, divisors: &[u64]) -> Vec<u64>;

    /// Dimension of the lifted/parallel representation for a given N.
    fn dimensionality(&self, n: u64) -> usize;

    /// Run both sequential and parallel, time them, and produce a report.
    fn report(&self, n: u64, divisors: &[u64]) -> DualityReport {
        let start_seq = Instant::now();
        let sequential_results = self.sequential(n, divisors);
        let sequential_time_ns = start_seq.elapsed().as_nanos() as u64;

        let start_par = Instant::now();
        let parallel_results = self.parallel(n, divisors);
        let parallel_time_ns = start_par.elapsed().as_nanos() as u64;

        let results_match = sequential_results == parallel_results;
        let speedup = if parallel_time_ns > 0 {
            sequential_time_ns as f64 / parallel_time_ns as f64
        } else {
            f64::INFINITY
        };

        DualityReport {
            strategy_name: self.name().to_string(),
            n,
            divisors_tested: divisors.len(),
            sequential_results,
            parallel_results,
            results_match,
            dimensionality: self.dimensionality(n),
            sequential_time_ns,
            parallel_time_ns,
            speedup,
        }
    }
}

/// Simple baseline: compute N mod d for each divisor d.
pub fn trial_division_sequential(n: u64, divisors: &[u64]) -> Vec<u64> {
    divisors.iter().map(|&d| n % d).collect()
}

/// Return all divisors where the corresponding residue is zero.
pub fn find_factors_from_residues(residues: &[u64], divisors: &[u64]) -> Vec<u64> {
    residues
        .iter()
        .zip(divisors.iter())
        .filter(|(&r, _)| r == 0)
        .map(|(_, &d)| d)
        .collect()
}

/// Generate all primes up to `limit` using a simple sieve.
pub fn primes_up_to(limit: u64) -> Vec<u64> {
    if limit < 2 {
        return vec![];
    }
    let n = limit as usize;
    let mut is_prime = vec![true; n + 1];
    is_prime[0] = false;
    is_prime[1] = false;
    let mut i = 2;
    while i * i <= n {
        if is_prime[i] {
            let mut j = i * i;
            while j <= n {
                is_prime[j] = false;
                j += i;
            }
        }
        i += 1;
    }
    is_prime
        .into_iter()
        .enumerate()
        .filter(|(_, p)| *p)
        .map(|(i, _)| i as u64)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::binary_lift::BinaryLift;
    use crate::crt_parallel::CrtParallel;
    use crate::ntt_domain::NttDomain;

    #[test]
    fn test_sequential_baseline() {
        let result = trial_division_sequential(35, &[2, 3, 5, 7]);
        assert_eq!(result, vec![1, 2, 0, 0]);
    }

    #[test]
    fn test_binary_lift_correctness() {
        let bl = BinaryLift;
        let divisors = vec![2, 3, 5, 7, 11];
        let seq = bl.sequential(77, &divisors);
        let par = bl.parallel(77, &divisors);
        assert_eq!(seq, par, "BinaryLift parallel must match sequential for n=77");
    }

    #[test]
    fn test_ntt_forward_inverse() {
        use crate::ntt_domain::{ntt_forward, ntt_inverse, find_primitive_root, mod_pow_u64};

        let modulus: u64 = 998244353;
        let g = find_primitive_root(modulus);
        let n = 8u64;
        // root of unity of order n: g^((modulus-1)/n)
        let root = mod_pow_u64(g, (modulus - 1) / n, modulus);

        let original: Vec<u64> = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let mut data = original.clone();

        ntt_forward(&mut data, modulus, root);
        // After forward NTT, data should differ from original (unless trivial)
        assert_ne!(data, original, "Forward NTT should change the data");

        ntt_inverse(&mut data, modulus, root);
        assert_eq!(data, original, "Inverse NTT should recover original data");
    }

    #[test]
    fn test_crt_reconstruct() {
        use crate::crt_parallel::crt_reconstruct;
        // 77 mod 3 = 2, 77 mod 5 = 2, 77 mod 7 = 0
        let result = crt_reconstruct(&[2, 2, 0], &[3, 5, 7]);
        assert_eq!(result, 77);
    }

    #[test]
    fn test_crt_parallel_correctness() {
        let crt = CrtParallel::new_default();
        let divisors = vec![2, 3, 5, 7, 11, 13];
        let seq = crt.sequential(143, &divisors);
        let par = crt.parallel(143, &divisors);
        assert_eq!(seq, par, "CrtParallel parallel must match sequential for n=143");
    }

    #[test]
    fn test_find_factors_from_residues() {
        let residues = vec![1, 2, 0, 0, 5];
        let divisors = vec![2, 3, 5, 7, 11];
        let factors = find_factors_from_residues(&residues, &divisors);
        assert_eq!(factors, vec![5, 7]);
    }

    #[test]
    fn test_ntt_domain_correctness() {
        let ntt = NttDomain::new_default();
        let divisors = vec![2, 3, 5, 7, 11];
        let seq = ntt.sequential(77, &divisors);
        let par = ntt.parallel(77, &divisors);
        assert_eq!(seq, par, "NttDomain parallel must match sequential for n=77");
    }

    #[test]
    fn test_primes_up_to() {
        assert_eq!(primes_up_to(20), vec![2, 3, 5, 7, 11, 13, 17, 19]);
        assert_eq!(primes_up_to(1), vec![]);
        assert_eq!(primes_up_to(2), vec![2]);
    }
}
