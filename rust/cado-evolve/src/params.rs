//! GNFS parameter space definition and genetic operators.
//!
//! Defines the tunable CADO-NFS parameters as a structured type with:
//! - Default parameter schedules for each bit size
//! - Random initialization within valid ranges
//! - Mutation (tweak 1-3 parameters by ±10-50%)
//! - Crossover (uniform blend of two configurations)
//! - Serialization to CADO-NFS CLI arguments

use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Complete set of tunable CADO-NFS parameters for GNFS.
///
/// These are the ~10 most impactful parameters that affect GNFS performance.
/// Each maps directly to a CADO-NFS command-line flag.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CadoParams {
    /// Polynomial degree for NFS polynomial selection.
    /// CADO flag: `tasks.polyselect.degree`
    /// Range: 3-6. Higher degrees better for larger N.
    pub poly_degree: u32,

    /// Maximum leading coefficient for polynomial search.
    /// CADO flag: `tasks.polyselect.admax`
    /// Range: 1e3-1e7. Larger = more time in polyselect, potentially better polynomial.
    pub poly_admax: u64,

    /// Increment step for leading coefficient search.
    /// CADO flag: `tasks.polyselect.incr`
    /// Range: 60-420. Must be divisible by 60.
    pub poly_incr: u64,

    /// Rational factor base bound.
    /// CADO flag: `tasks.sieve.rlim`
    /// Range: 1e5-1e8. Primes up to this bound in rational factor base.
    pub fb_rational_bound: u64,

    /// Algebraic factor base bound.
    /// CADO flag: `tasks.sieve.alim`
    /// Range: 1e5-1e8. Primes up to this bound in algebraic factor base.
    pub fb_algebraic_bound: u64,

    /// Rational large prime bits.
    /// CADO flag: `tasks.sieve.lpbr`
    /// Range: 20-32. Max size (in bits) of large primes on rational side.
    pub lp_rational_bits: u32,

    /// Algebraic large prime bits.
    /// CADO flag: `tasks.sieve.lpba`
    /// Range: 20-32. Max size (in bits) of large primes on algebraic side.
    pub lp_algebraic_bits: u32,

    /// Rational cofactor bound (bits).
    /// CADO flag: `tasks.sieve.mfbr`
    /// Range: 40-64. Maximum bits for cofactors on rational side.
    pub sieve_mfbr: u32,

    /// Algebraic cofactor bound (bits).
    /// CADO flag: `tasks.sieve.mfba`
    /// Range: 40-64. Maximum bits for cofactors on algebraic side.
    pub sieve_mfba: u32,

    /// Special-q range per workunit.
    /// CADO flag: `tasks.sieve.qrange`
    /// Range: 1e3-1e6. Number of special-q values per sieve workunit.
    pub sieve_qrange: u64,
}

/// Valid ranges for each parameter.
struct ParamRanges {
    poly_degree: (u32, u32),
    poly_admax: (u64, u64),
    poly_incr: (u64, u64),
    fb_rational_bound: (u64, u64),
    fb_algebraic_bound: (u64, u64),
    lp_rational_bits: (u32, u32),
    lp_algebraic_bits: (u32, u32),
    sieve_mfbr: (u32, u32),
    sieve_mfba: (u32, u32),
    sieve_qrange: (u64, u64),
}

impl ParamRanges {
    /// Get valid ranges for the given bit size.
    fn for_bits(n_bits: u32) -> Self {
        if n_bits <= 100 {
            ParamRanges {
                poly_degree: (3, 4),
                poly_admax: (1_000, 100_000),
                poly_incr: (60, 300),
                fb_rational_bound: (100_000, 5_000_000),
                fb_algebraic_bound: (100_000, 5_000_000),
                lp_rational_bits: (20, 26),
                lp_algebraic_bits: (20, 26),
                sieve_mfbr: (40, 52),
                sieve_mfba: (40, 52),
                sieve_qrange: (1_000, 100_000),
            }
        } else if n_bits <= 150 {
            ParamRanges {
                poly_degree: (4, 5),
                poly_admax: (10_000, 1_000_000),
                poly_incr: (60, 420),
                fb_rational_bound: (500_000, 20_000_000),
                fb_algebraic_bound: (500_000, 20_000_000),
                lp_rational_bits: (22, 28),
                lp_algebraic_bits: (22, 28),
                sieve_mfbr: (44, 56),
                sieve_mfba: (44, 56),
                sieve_qrange: (5_000, 500_000),
            }
        } else {
            ParamRanges {
                poly_degree: (5, 6),
                poly_admax: (100_000, 10_000_000),
                poly_incr: (60, 420),
                fb_rational_bound: (1_000_000, 100_000_000),
                fb_algebraic_bound: (1_000_000, 100_000_000),
                lp_rational_bits: (24, 32),
                lp_algebraic_bits: (24, 32),
                sieve_mfbr: (48, 64),
                sieve_mfba: (48, 64),
                sieve_qrange: (10_000, 1_000_000),
            }
        }
    }
}

impl CadoParams {
    /// CADO-NFS default parameters for a given bit size.
    ///
    /// These approximate the built-in CADO-NFS parameter tables.
    /// See `parameters/factor/params.cXX` in the CADO-NFS source tree.
    pub fn default_for_bits(n_bits: u32) -> Self {
        match n_bits {
            0..=80 => CadoParams {
                poly_degree: 3,
                poly_admax: 2_000,
                poly_incr: 60,
                fb_rational_bound: 200_000,
                fb_algebraic_bound: 200_000,
                lp_rational_bits: 21,
                lp_algebraic_bits: 21,
                sieve_mfbr: 42,
                sieve_mfba: 42,
                sieve_qrange: 2_000,
            },
            81..=100 => CadoParams {
                poly_degree: 3,
                poly_admax: 10_000,
                poly_incr: 60,
                fb_rational_bound: 500_000,
                fb_algebraic_bound: 500_000,
                lp_rational_bits: 22,
                lp_algebraic_bits: 22,
                sieve_mfbr: 44,
                sieve_mfba: 44,
                sieve_qrange: 5_000,
            },
            101..=120 => CadoParams {
                poly_degree: 4,
                poly_admax: 50_000,
                poly_incr: 60,
                fb_rational_bound: 2_000_000,
                fb_algebraic_bound: 2_000_000,
                lp_rational_bits: 24,
                lp_algebraic_bits: 24,
                sieve_mfbr: 48,
                sieve_mfba: 48,
                sieve_qrange: 10_000,
            },
            121..=140 => CadoParams {
                poly_degree: 4,
                poly_admax: 200_000,
                poly_incr: 120,
                fb_rational_bound: 5_000_000,
                fb_algebraic_bound: 5_000_000,
                lp_rational_bits: 25,
                lp_algebraic_bits: 25,
                sieve_mfbr: 50,
                sieve_mfba: 50,
                sieve_qrange: 50_000,
            },
            141..=160 => CadoParams {
                poly_degree: 5,
                poly_admax: 500_000,
                poly_incr: 120,
                fb_rational_bound: 10_000_000,
                fb_algebraic_bound: 10_000_000,
                lp_rational_bits: 27,
                lp_algebraic_bits: 27,
                sieve_mfbr: 54,
                sieve_mfba: 54,
                sieve_qrange: 100_000,
            },
            161..=180 => CadoParams {
                poly_degree: 5,
                poly_admax: 2_000_000,
                poly_incr: 180,
                fb_rational_bound: 30_000_000,
                fb_algebraic_bound: 30_000_000,
                lp_rational_bits: 28,
                lp_algebraic_bits: 28,
                sieve_mfbr: 56,
                sieve_mfba: 56,
                sieve_qrange: 200_000,
            },
            _ => CadoParams {
                poly_degree: 5,
                poly_admax: 5_000_000,
                poly_incr: 240,
                fb_rational_bound: 50_000_000,
                fb_algebraic_bound: 50_000_000,
                lp_rational_bits: 30,
                lp_algebraic_bits: 30,
                sieve_mfbr: 60,
                sieve_mfba: 60,
                sieve_qrange: 500_000,
            },
        }
    }

    /// Generate a random valid parameter configuration for the given bit size.
    pub fn random(rng: &mut impl Rng, n_bits: u32) -> Self {
        let ranges = ParamRanges::for_bits(n_bits);

        CadoParams {
            poly_degree: rng.gen_range(ranges.poly_degree.0..=ranges.poly_degree.1),
            poly_admax: log_uniform(rng, ranges.poly_admax.0, ranges.poly_admax.1),
            poly_incr: round_to_60(rng.gen_range(ranges.poly_incr.0..=ranges.poly_incr.1)),
            fb_rational_bound: log_uniform(
                rng,
                ranges.fb_rational_bound.0,
                ranges.fb_rational_bound.1,
            ),
            fb_algebraic_bound: log_uniform(
                rng,
                ranges.fb_algebraic_bound.0,
                ranges.fb_algebraic_bound.1,
            ),
            lp_rational_bits: rng
                .gen_range(ranges.lp_rational_bits.0..=ranges.lp_rational_bits.1),
            lp_algebraic_bits: rng
                .gen_range(ranges.lp_algebraic_bits.0..=ranges.lp_algebraic_bits.1),
            sieve_mfbr: rng.gen_range(ranges.sieve_mfbr.0..=ranges.sieve_mfbr.1),
            sieve_mfba: rng.gen_range(ranges.sieve_mfba.0..=ranges.sieve_mfba.1),
            sieve_qrange: log_uniform(rng, ranges.sieve_qrange.0, ranges.sieve_qrange.1),
        }
    }

    /// Mutate this configuration by tweaking 1-3 parameters.
    ///
    /// Each selected parameter is adjusted by ±10-50% (multiplicative).
    /// The result is clamped to valid ranges.
    pub fn mutate(&self, rng: &mut impl Rng, n_bits: u32) -> Self {
        let mut child = self.clone();
        let ranges = ParamRanges::for_bits(n_bits);
        let num_mutations = rng.gen_range(1..=3);

        for _ in 0..num_mutations {
            let param_idx = rng.gen_range(0..10);
            let factor = 1.0 + rng.gen_range(-0.5..0.5_f64);

            match param_idx {
                0 => {
                    // poly_degree: ±1
                    let delta: i32 = if rng.gen_bool(0.5) { 1 } else { -1 };
                    child.poly_degree =
                        (child.poly_degree as i32 + delta).clamp(
                            ranges.poly_degree.0 as i32,
                            ranges.poly_degree.1 as i32,
                        ) as u32;
                }
                1 => {
                    child.poly_admax = ((child.poly_admax as f64 * factor) as u64)
                        .clamp(ranges.poly_admax.0, ranges.poly_admax.1);
                }
                2 => {
                    child.poly_incr = round_to_60(
                        ((child.poly_incr as f64 * factor) as u64)
                            .clamp(ranges.poly_incr.0, ranges.poly_incr.1),
                    );
                }
                3 => {
                    child.fb_rational_bound = ((child.fb_rational_bound as f64 * factor) as u64)
                        .clamp(ranges.fb_rational_bound.0, ranges.fb_rational_bound.1);
                }
                4 => {
                    child.fb_algebraic_bound = ((child.fb_algebraic_bound as f64 * factor) as u64)
                        .clamp(ranges.fb_algebraic_bound.0, ranges.fb_algebraic_bound.1);
                }
                5 => {
                    let delta: i32 = rng.gen_range(-2..=2);
                    child.lp_rational_bits = (child.lp_rational_bits as i32 + delta).clamp(
                        ranges.lp_rational_bits.0 as i32,
                        ranges.lp_rational_bits.1 as i32,
                    ) as u32;
                }
                6 => {
                    let delta: i32 = rng.gen_range(-2..=2);
                    child.lp_algebraic_bits = (child.lp_algebraic_bits as i32 + delta).clamp(
                        ranges.lp_algebraic_bits.0 as i32,
                        ranges.lp_algebraic_bits.1 as i32,
                    ) as u32;
                }
                7 => {
                    let delta: i32 = rng.gen_range(-4..=4);
                    child.sieve_mfbr = (child.sieve_mfbr as i32 + delta)
                        .clamp(ranges.sieve_mfbr.0 as i32, ranges.sieve_mfbr.1 as i32)
                        as u32;
                }
                8 => {
                    let delta: i32 = rng.gen_range(-4..=4);
                    child.sieve_mfba = (child.sieve_mfba as i32 + delta)
                        .clamp(ranges.sieve_mfba.0 as i32, ranges.sieve_mfba.1 as i32)
                        as u32;
                }
                _ => {
                    child.sieve_qrange = ((child.sieve_qrange as f64 * factor) as u64)
                        .clamp(ranges.sieve_qrange.0, ranges.sieve_qrange.1);
                }
            }
        }

        // Enforce constraint: mfbr >= 2 * lpbr, mfba >= 2 * lpba
        child.sieve_mfbr = child.sieve_mfbr.max(2 * child.lp_rational_bits);
        child.sieve_mfba = child.sieve_mfba.max(2 * child.lp_algebraic_bits);

        child
    }

    /// Crossover (uniform blend) of two parameter configurations.
    ///
    /// Each parameter is independently drawn from one of the two parents.
    pub fn crossover(&self, other: &Self, rng: &mut impl Rng) -> Self {
        // Generate a bitmask of parent selections upfront to avoid borrow issues
        let mask: u16 = rng.gen();

        let mut child = CadoParams {
            poly_degree: if mask & (1 << 0) != 0 { self.poly_degree } else { other.poly_degree },
            poly_admax: if mask & (1 << 1) != 0 { self.poly_admax } else { other.poly_admax },
            poly_incr: if mask & (1 << 2) != 0 { self.poly_incr } else { other.poly_incr },
            fb_rational_bound: if mask & (1 << 3) != 0 { self.fb_rational_bound } else { other.fb_rational_bound },
            fb_algebraic_bound: if mask & (1 << 4) != 0 { self.fb_algebraic_bound } else { other.fb_algebraic_bound },
            lp_rational_bits: if mask & (1 << 5) != 0 { self.lp_rational_bits } else { other.lp_rational_bits },
            lp_algebraic_bits: if mask & (1 << 6) != 0 { self.lp_algebraic_bits } else { other.lp_algebraic_bits },
            sieve_mfbr: if mask & (1 << 7) != 0 { self.sieve_mfbr } else { other.sieve_mfbr },
            sieve_mfba: if mask & (1 << 8) != 0 { self.sieve_mfba } else { other.sieve_mfba },
            sieve_qrange: if mask & (1 << 9) != 0 { self.sieve_qrange } else { other.sieve_qrange },
        };

        // Enforce constraint: mfbr >= 2 * lpbr, mfba >= 2 * lpba
        child.sieve_mfbr = child.sieve_mfbr.max(2 * child.lp_rational_bits);
        child.sieve_mfba = child.sieve_mfba.max(2 * child.lp_algebraic_bits);

        child
    }

    /// Convert parameters to CADO-NFS command-line arguments.
    pub fn to_cado_args(&self) -> Vec<String> {
        // Parameter names must match CADO-NFS's actual naming convention:
        //   tasks.lim0/lim1    = factor base bounds (rational/algebraic)
        //   tasks.lpb0/lpb1    = large prime bits
        //   tasks.sieve.mfb0/1 = cofactor bounds
        vec![
            format!("tasks.polyselect.degree={}", self.poly_degree),
            format!("tasks.polyselect.admax={}", self.poly_admax),
            format!("tasks.polyselect.incr={}", self.poly_incr),
            format!("tasks.lim0={}", self.fb_rational_bound),
            format!("tasks.lim1={}", self.fb_algebraic_bound),
            format!("tasks.lpb0={}", self.lp_rational_bits),
            format!("tasks.lpb1={}", self.lp_algebraic_bits),
            format!("tasks.sieve.mfb0={}", self.sieve_mfbr),
            format!("tasks.sieve.mfb1={}", self.sieve_mfba),
            format!("tasks.sieve.qrange={}", self.sieve_qrange),
        ]
    }

    /// Write parameters to a CADO-NFS parameter file.
    pub fn to_param_file(&self, path: &std::path::Path) -> std::io::Result<()> {
        let content = format!(
            "# Auto-generated by cado-evolve\n\
             tasks.polyselect.degree = {}\n\
             tasks.polyselect.admax = {}\n\
             tasks.polyselect.incr = {}\n\
             tasks.lim0 = {}\n\
             tasks.lim1 = {}\n\
             tasks.lpb0 = {}\n\
             tasks.lpb1 = {}\n\
             tasks.sieve.mfb0 = {}\n\
             tasks.sieve.mfb1 = {}\n\
             tasks.sieve.qrange = {}\n",
            self.poly_degree,
            self.poly_admax,
            self.poly_incr,
            self.fb_rational_bound,
            self.fb_algebraic_bound,
            self.lp_rational_bits,
            self.lp_algebraic_bits,
            self.sieve_mfbr,
            self.sieve_mfba,
            self.sieve_qrange,
        );
        std::fs::write(path, content)
    }

    /// Compute a hash for caching fitness evaluations.
    pub fn fitness_hash(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.poly_degree.hash(&mut hasher);
        self.poly_admax.hash(&mut hasher);
        self.poly_incr.hash(&mut hasher);
        self.fb_rational_bound.hash(&mut hasher);
        self.fb_algebraic_bound.hash(&mut hasher);
        self.lp_rational_bits.hash(&mut hasher);
        self.lp_algebraic_bits.hash(&mut hasher);
        self.sieve_mfbr.hash(&mut hasher);
        self.sieve_mfba.hash(&mut hasher);
        self.sieve_qrange.hash(&mut hasher);
        hasher.finish()
    }

    /// Format parameters as a compact summary string.
    pub fn summary(&self) -> String {
        format!(
            "deg={} ad={} rlim={} alim={} lpb={}/{} mfb={}/{} qr={}",
            self.poly_degree,
            self.poly_admax,
            self.fb_rational_bound,
            self.fb_algebraic_bound,
            self.lp_rational_bits,
            self.lp_algebraic_bits,
            self.sieve_mfbr,
            self.sieve_mfba,
            self.sieve_qrange,
        )
    }
}

impl std::fmt::Display for CadoParams {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "CadoParams {{ degree={}, admax={}, incr={}, rlim={}, alim={}, lpbr={}, lpba={}, mfbr={}, mfba={}, qrange={} }}",
            self.poly_degree,
            self.poly_admax,
            self.poly_incr,
            self.fb_rational_bound,
            self.fb_algebraic_bound,
            self.lp_rational_bits,
            self.lp_algebraic_bits,
            self.sieve_mfbr,
            self.sieve_mfba,
            self.sieve_qrange,
        )
    }
}

/// Sample from a log-uniform distribution between min and max.
///
/// This is appropriate for parameters that span orders of magnitude
/// (e.g., factor base bounds from 1e5 to 1e8).
fn log_uniform(rng: &mut impl Rng, min: u64, max: u64) -> u64 {
    let log_min = (min as f64).ln();
    let log_max = (max as f64).ln();
    let log_val = rng.gen_range(log_min..log_max);
    log_val.exp() as u64
}

/// Round a value to the nearest multiple of 60.
fn round_to_60(val: u64) -> u64 {
    ((val + 30) / 60) * 60
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::thread_rng;

    #[test]
    fn test_default_params_100bit() {
        let params = CadoParams::default_for_bits(100);
        assert_eq!(params.poly_degree, 3);
        assert!(params.fb_rational_bound > 0);
        assert!(params.lp_rational_bits >= 20);
        assert!(params.sieve_mfbr >= 2 * params.lp_rational_bits);
    }

    #[test]
    fn test_default_params_150bit() {
        let params = CadoParams::default_for_bits(150);
        assert!(params.poly_degree >= 4);
        assert!(params.fb_rational_bound > 1_000_000);
    }

    #[test]
    fn test_default_params_200bit() {
        let params = CadoParams::default_for_bits(200);
        assert!(params.poly_degree >= 5);
    }

    #[test]
    fn test_random_params() {
        let mut rng = thread_rng();
        for bits in [80, 100, 120, 150, 180, 200] {
            let params = CadoParams::random(&mut rng, bits);
            assert!(params.poly_degree >= 3 && params.poly_degree <= 6);
            assert!(params.fb_rational_bound > 0);
            assert!(params.lp_rational_bits >= 20 && params.lp_rational_bits <= 32);
            assert_eq!(params.poly_incr % 60, 0, "poly_incr must be multiple of 60");
        }
    }

    #[test]
    fn test_mutation_stays_in_range() {
        let mut rng = thread_rng();
        let base = CadoParams::default_for_bits(120);

        for _ in 0..100 {
            let mutated = base.mutate(&mut rng, 120);
            assert!(mutated.poly_degree >= 3 && mutated.poly_degree <= 6);
            assert!(mutated.lp_rational_bits >= 20 && mutated.lp_rational_bits <= 32);
            assert!(mutated.sieve_mfbr >= 2 * mutated.lp_rational_bits);
            assert!(mutated.sieve_mfba >= 2 * mutated.lp_algebraic_bits);
        }
    }

    #[test]
    fn test_crossover() {
        let mut rng = thread_rng();
        let parent_a = CadoParams::default_for_bits(100);
        let parent_b = CadoParams::default_for_bits(150);

        let child = parent_a.crossover(&parent_b, &mut rng);
        // Child should have values from either parent
        assert!(
            child.poly_degree == parent_a.poly_degree
                || child.poly_degree == parent_b.poly_degree
        );
        // Constraint must hold
        assert!(child.sieve_mfbr >= 2 * child.lp_rational_bits);
    }

    #[test]
    fn test_to_cado_args() {
        let params = CadoParams::default_for_bits(100);
        let args = params.to_cado_args();
        assert_eq!(args.len(), 10);
        assert!(args[0].starts_with("tasks.polyselect.degree="));
        // CADO-NFS uses tasks.lim0/lim1 (not tasks.sieve.rlim/alim)
        assert!(args[3].starts_with("tasks.lim0="));
        assert!(args[4].starts_with("tasks.lim1="));
        assert!(args[5].starts_with("tasks.lpb0="));
        assert!(args[6].starts_with("tasks.lpb1="));
        assert!(args[7].starts_with("tasks.sieve.mfb0="));
        assert!(args[8].starts_with("tasks.sieve.mfb1="));
    }

    #[test]
    fn test_to_param_file() {
        let params = CadoParams::default_for_bits(100);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.params");
        params.to_param_file(&path).unwrap();

        let content = std::fs::read_to_string(&path).unwrap();
        assert!(content.contains("tasks.polyselect.degree"));
        assert!(content.contains("tasks.lim0"));
        assert!(content.contains("tasks.lpb0"));
        assert!(content.contains("tasks.sieve.mfb0"));
    }

    #[test]
    fn test_fitness_hash_deterministic() {
        let params = CadoParams::default_for_bits(100);
        let h1 = params.fitness_hash();
        let h2 = params.fitness_hash();
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_fitness_hash_different_params() {
        let a = CadoParams::default_for_bits(100);
        let b = CadoParams::default_for_bits(150);
        assert_ne!(a.fitness_hash(), b.fitness_hash());
    }

    #[test]
    fn test_round_to_60() {
        assert_eq!(round_to_60(60), 60);
        assert_eq!(round_to_60(89), 60);   // 89 rounds down to 60
        assert_eq!(round_to_60(90), 120);  // 90 rounds up to 120
        assert_eq!(round_to_60(91), 120);
        assert_eq!(round_to_60(120), 120);
        assert_eq!(round_to_60(59), 60);
        assert_eq!(round_to_60(0), 0);
        assert_eq!(round_to_60(30), 60);   // 30 rounds up to 60
    }

    #[test]
    fn test_display() {
        let params = CadoParams::default_for_bits(100);
        let s = format!("{}", params);
        assert!(s.contains("degree=3"));
        assert!(s.contains("rlim="));
    }

    #[test]
    fn test_summary() {
        let params = CadoParams::default_for_bits(100);
        let s = params.summary();
        assert!(s.contains("deg=3"));
        assert!(s.contains("lpb="));
    }
}
