/// E19: Eisenstein Congruence Hunt
///
/// Exhaustive fail-fast search for poly(log N)-computable functions
/// matching σ_{k-1}(N) mod ℓ across Eisenstein congruence channels.

pub mod arith;
pub mod candidates;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::collections::HashMap;

/// A congruence channel: (weight k, Bernoulli prime ℓ).
#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct Channel {
    pub weight: u32,
    pub ell: u64,
}

/// All 7 Eisenstein congruence channels from E13.
pub const CHANNELS: &[Channel] = &[
    Channel { weight: 12, ell: 691 },
    Channel { weight: 16, ell: 3617 },
    Channel { weight: 18, ell: 43867 },
    Channel { weight: 20, ell: 283 },
    Channel { weight: 20, ell: 617 },
    Channel { weight: 22, ell: 131 },
    Channel { weight: 22, ell: 593 },
];

/// A balanced semiprime with known factors.
#[derive(Debug, Clone, Copy)]
pub struct Semiprime {
    pub n: u64,
    pub p: u64,
    pub q: u64,
}

/// Ground truth: σ_{k-1}(N) mod ℓ for N = pq.
/// σ_{k-1}(N) = 1 + p^{k-1} + q^{k-1} + N^{k-1} since divisors of pq are {1, p, q, pq}.
pub fn ground_truth(sp: &Semiprime, ch: &Channel) -> u64 {
    let ell = ch.ell;
    let k1 = (ch.weight - 1) as u64;
    let pk = arith::mod_pow(sp.p, k1, ell);
    let qk = arith::mod_pow(sp.q, k1, ell);
    let nk = arith::mod_pow(sp.n, k1, ell);
    (1 + pk + qk + nk) % ell
}

/// Generate balanced semiprimes with bit sizes in [min_bits, max_bits].
/// Uses a fixed seed for reproducibility. Deduplicates by N value.
pub fn generate_semiprimes(count: usize, min_bits: u32, max_bits: u32, seed: u64) -> Vec<Semiprime> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut result = Vec::with_capacity(count);
    let mut seen = std::collections::HashSet::with_capacity(count);

    while result.len() < count {
        let total_bits = rng.gen_range(min_bits..=max_bits);
        let half = total_bits / 2;
        if half < 2 {
            continue;
        }

        let p = arith::random_prime(half, &mut rng);
        let q = arith::random_prime(half, &mut rng);

        if p == q {
            continue;
        }

        let (small, big) = if p < q { (p, q) } else { (q, p) };

        // Balance: small/big >= 0.3
        if (small as f64) / (big as f64) < 0.3 {
            continue;
        }

        let n = small as u128 * big as u128;
        if n > u64::MAX as u128 {
            continue;
        }

        let n64 = n as u64;
        if !seen.insert(n64) {
            continue; // skip duplicate N
        }

        result.push(Semiprime {
            n: n64,
            p: small,
            q: big,
        });
    }

    result
}

/// Check if σ_{k-1}(N) mod ℓ is a function of (N mod ℓ) alone.
/// If true for all semiprimes, some polynomial over F_ℓ could match.
/// Returns (is_consistent, num_collisions_tested).
pub fn check_collision(ch: &Channel, semiprimes: &[Semiprime], targets: &[u64]) -> (bool, usize) {
    let mut map: HashMap<u64, u64> = HashMap::new();
    let mut collisions = 0usize;

    for (sp, &target) in semiprimes.iter().zip(targets.iter()) {
        let n_mod = sp.n % ch.ell;
        match map.entry(n_mod) {
            std::collections::hash_map::Entry::Occupied(e) => {
                collisions += 1;
                if *e.get() != target {
                    return (false, collisions);
                }
            }
            std::collections::hash_map::Entry::Vacant(e) => {
                e.insert(target);
            }
        }
    }

    (true, collisions)
}

/// Check if σ_{k-1}(N) mod ℓ is a function of (N mod ℓ²) alone.
/// Stronger than mod-ℓ check: covers functions depending on the first two ℓ-adic digits.
/// Returns (is_consistent, num_collisions_tested).
pub fn check_collision_sq(ch: &Channel, semiprimes: &[Semiprime], targets: &[u64]) -> (bool, usize) {
    let ell_sq = ch.ell * ch.ell;
    let mut map: HashMap<u64, u64> = HashMap::new();
    let mut collisions = 0usize;

    for (sp, &target) in semiprimes.iter().zip(targets.iter()) {
        let n_mod = sp.n % ell_sq;
        match map.entry(n_mod) {
            std::collections::hash_map::Entry::Occupied(e) => {
                collisions += 1;
                if *e.get() != target {
                    return (false, collisions);
                }
            }
            std::collections::hash_map::Entry::Vacant(e) => {
                e.insert(target);
            }
        }
    }

    (true, collisions)
}

/// Check if σ_{k-1}(N) mod ℓ is a function of (N mod m) alone, for arbitrary modulus m.
/// Generalizes check_collision to any modulus, not just ℓ or ℓ².
pub fn check_collision_aux(semiprimes: &[Semiprime], targets: &[u64], modulus: u64) -> (bool, usize) {
    let mut map: HashMap<u64, u64> = HashMap::new();
    let mut collisions = 0usize;

    for (sp, &target) in semiprimes.iter().zip(targets.iter()) {
        let n_mod = sp.n % modulus;
        match map.entry(n_mod) {
            std::collections::hash_map::Entry::Occupied(e) => {
                collisions += 1;
                if *e.get() != target {
                    return (false, collisions);
                }
            }
            std::collections::hash_map::Entry::Vacant(e) => {
                e.insert(target);
            }
        }
    }

    (true, collisions)
}

/// Check if σ_{k-1}(N) mod ℓ is a function of (N mod m1, N mod m2) jointly.
/// By CRT, equivalent to N mod lcm(m1, m2) when gcd(m1, m2) = 1.
/// Stronger than individual mod-m checks: covers all functions of both residues.
pub fn check_collision_joint(semiprimes: &[Semiprime], targets: &[u64], m1: u64, m2: u64) -> (bool, usize) {
    let mut map: HashMap<(u64, u64), u64> = HashMap::new();
    let mut collisions = 0usize;

    for (sp, &target) in semiprimes.iter().zip(targets.iter()) {
        let key = (sp.n % m1, sp.n % m2);
        match map.entry(key) {
            std::collections::hash_map::Entry::Occupied(e) => {
                collisions += 1;
                if *e.get() != target {
                    return (false, collisions);
                }
            }
            std::collections::hash_map::Entry::Vacant(e) => {
                e.insert(target);
            }
        }
    }

    (true, collisions)
}

/// Result of a collision check against an auxiliary modulus.
#[derive(Debug, serde::Serialize)]
pub struct AuxCollisionResult {
    pub label: String,
    pub modulus: u64,
    pub consistent: bool,
    pub collisions_tested: usize,
}

/// Per-channel search result.
#[derive(Debug, serde::Serialize)]
pub struct ChannelResult {
    pub weight: u32,
    pub ell: u64,
    pub total_candidates: usize,
    pub survived_first: usize,
    pub survived_all: usize,
    pub survivors: Vec<String>,
    pub collision_consistent: bool,
    pub collision_tests: usize,
    pub collision_sq_consistent: bool,
    pub collision_sq_tests: usize,
    pub aux_collisions: Vec<AuxCollisionResult>,
}

/// Overall search result.
#[derive(Debug, serde::Serialize)]
pub struct SearchResult {
    pub channels: Vec<ChannelResult>,
    pub total_candidates_tested: usize,
    pub total_wall_seconds: f64,
    pub candidates_per_second: f64,
    pub num_semiprimes: usize,
    pub breakthrough: bool,
}

/// Run the fail-fast search for a single channel.
pub fn search_channel(
    ch: &Channel,
    semiprimes: &[Semiprime],
    targets: &[u64],
) -> ChannelResult {
    let all_candidates = candidates::generate_all(ch.ell);
    let total = all_candidates.len();

    // Rayon: parallelize over candidates, sequential fail-fast per candidate
    let survivors: Vec<(String, usize)> = all_candidates
        .into_par_iter()
        .filter_map(|cand| {
            let mut survived = 0usize;
            for (sp, &target) in semiprimes.iter().zip(targets.iter()) {
                let val = candidates::eval(&cand.kind, sp.n, ch.ell);
                if val != target {
                    return if survived >= 1 {
                        Some((cand.name, survived))
                    } else {
                        None
                    };
                }
                survived += 1;
            }
            Some((cand.name, survived))
        })
        .collect();

    let survived_first = survivors.len();
    let num_sp = semiprimes.len();
    let survived_all = survivors.iter().filter(|(_, c)| *c == num_sp).count();
    let full_survivors: Vec<String> = survivors
        .iter()
        .filter(|(_, c)| *c == num_sp)
        .map(|(name, _)| name.clone())
        .collect();

    let (collision_consistent, collision_tests) = check_collision(ch, semiprimes, targets);
    let (collision_sq_consistent, collision_sq_tests) = check_collision_sq(ch, semiprimes, targets);

    // Auxiliary modulus collision checks
    let mut aux_collisions = Vec::new();

    // N mod (ℓ-1): tests all functions depending on multiplicative order structure
    let (c, t) = check_collision_aux(semiprimes, targets, ch.ell - 1);
    aux_collisions.push(AuxCollisionResult {
        label: format!("N mod (ℓ-1={}))", ch.ell - 1),
        modulus: ch.ell - 1,
        consistent: c,
        collisions_tested: t,
    });

    // N mod (ℓ+1): tests Frobenius trace analog
    let (c, t) = check_collision_aux(semiprimes, targets, ch.ell + 1);
    aux_collisions.push(AuxCollisionResult {
        label: format!("N mod (ℓ+1={})", ch.ell + 1),
        modulus: ch.ell + 1,
        consistent: c,
        collisions_tested: t,
    });

    // N mod 2ℓ: tests parity + mod ℓ
    let (c, t) = check_collision_aux(semiprimes, targets, 2 * ch.ell);
    aux_collisions.push(AuxCollisionResult {
        label: format!("N mod 2ℓ={}", 2 * ch.ell),
        modulus: 2 * ch.ell,
        consistent: c,
        collisions_tested: t,
    });

    // Joint: (N mod ℓ, N mod (ℓ-1)) — strongest single-channel test
    // By CRT (gcd(ℓ, ℓ-1) = 1), equivalent to N mod ℓ(ℓ-1)
    let (c, t) = check_collision_joint(semiprimes, targets, ch.ell, ch.ell - 1);
    aux_collisions.push(AuxCollisionResult {
        label: format!("(N mod ℓ, N mod ℓ-1) joint"),
        modulus: ch.ell * (ch.ell - 1),
        consistent: c,
        collisions_tested: t,
    });

    ChannelResult {
        weight: ch.weight,
        ell: ch.ell,
        total_candidates: total,
        survived_first,
        survived_all,
        survivors: full_survivors,
        collision_consistent,
        collision_tests,
        collision_sq_consistent,
        collision_sq_tests,
        aux_collisions,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ground_truth_n15() {
        let sp = Semiprime { n: 15, p: 3, q: 5 };
        let ch = Channel { weight: 12, ell: 691 };
        let gt = ground_truth(&sp, &ch);
        // Manual: 1 + 3^11 + 5^11 + 15^11 mod 691
        let expected = (1
            + arith::mod_pow(3, 11, 691)
            + arith::mod_pow(5, 11, 691)
            + arith::mod_pow(15, 11, 691))
            % 691;
        assert_eq!(gt, expected);
    }

    #[test]
    fn test_ground_truth_multiplicative() {
        // σ_{k-1}(pq) = (1 + p^{k-1})(1 + q^{k-1}) for multiplicative σ
        let sp = Semiprime { n: 77, p: 7, q: 11 };
        let ch = Channel { weight: 12, ell: 691 };
        let gt = ground_truth(&sp, &ch);
        let factor_product = ((1 + arith::mod_pow(7, 11, 691)) as u128
            * (1 + arith::mod_pow(11, 11, 691)) as u128
            % 691 as u128) as u64;
        assert_eq!(gt, factor_product);
    }

    #[test]
    fn test_generate_semiprimes() {
        let sps = generate_semiprimes(100, 16, 24, 42);
        assert_eq!(sps.len(), 100);
        for sp in &sps {
            assert_eq!(sp.n, sp.p * sp.q);
            assert!(sp.p < sp.q);
            assert!((sp.p as f64) / (sp.q as f64) >= 0.3);
            assert!(arith::is_prime_u64(sp.p));
            assert!(arith::is_prime_u64(sp.q));
        }
    }

    #[test]
    fn test_collision_check_false() {
        // σ_{k-1}(N) mod ℓ should NOT be a function of N mod ℓ alone
        let sps = generate_semiprimes(500, 16, 24, 42);
        let ch = Channel { weight: 22, ell: 131 }; // small ℓ = many collisions
        let targets: Vec<u64> = sps.iter().map(|sp| ground_truth(sp, &ch)).collect();
        let (consistent, collisions) = check_collision(&ch, &sps, &targets);
        assert!(
            !consistent,
            "expected collision inconsistency for ℓ=131 with 500 semiprimes (tested {} collisions)",
            collisions
        );
    }

    #[test]
    fn test_aux_collision_fails() {
        // σ_{k-1}(N) mod ℓ should NOT be a function of N mod (ℓ-1)
        let sps = generate_semiprimes(500, 16, 24, 42);
        let ch = Channel { weight: 22, ell: 131 };
        let targets: Vec<u64> = sps.iter().map(|sp| ground_truth(sp, &ch)).collect();
        let (consistent, collisions) = check_collision_aux(&sps, &targets, ch.ell - 1);
        assert!(
            !consistent,
            "expected N mod (ℓ-1) inconsistency for ℓ=131 with 500 semiprimes (tested {} collisions)",
            collisions
        );
    }

    #[test]
    fn test_joint_collision_fails() {
        // σ_{k-1}(N) mod ℓ should NOT be a function of (N mod ℓ, N mod (ℓ-1))
        let sps = generate_semiprimes(1000, 16, 24, 42);
        let ch = Channel { weight: 22, ell: 131 };
        let targets: Vec<u64> = sps.iter().map(|sp| ground_truth(sp, &ch)).collect();
        let (consistent, collisions) = check_collision_joint(&sps, &targets, ch.ell, ch.ell - 1);
        assert!(
            !consistent,
            "expected joint (ℓ, ℓ-1) inconsistency for ℓ=131 with 1000 semiprimes (tested {} collisions)",
            collisions
        );
    }

    #[test]
    fn test_no_power_residue_survives() {
        let sps = generate_semiprimes(10, 16, 20, 42);
        let ch = Channel { weight: 12, ell: 691 };
        let targets: Vec<u64> = sps.iter().map(|sp| ground_truth(sp, &ch)).collect();

        for a in 1..691u64 {
            let first_val = arith::mod_pow(sps[0].n % ch.ell, a, ch.ell);
            if first_val != targets[0] {
                continue;
            }
            let all_match = sps.iter().zip(targets.iter()).all(|(sp, &t)| {
                arith::mod_pow(sp.n % ch.ell, a, ch.ell) == t
            });
            assert!(
                !all_match,
                "power residue N^{} should not match σ_11 for all semiprimes",
                a
            );
        }
    }

    #[test]
    fn test_search_channel_no_survivors() {
        let sps = generate_semiprimes(50, 16, 20, 42);
        let ch = Channel { weight: 12, ell: 691 };
        let targets: Vec<u64> = sps.iter().map(|sp| ground_truth(sp, &ch)).collect();
        let result = search_channel(&ch, &sps, &targets);
        assert_eq!(
            result.survived_all, 0,
            "no candidates should survive all 50 semiprimes"
        );
    }
}
