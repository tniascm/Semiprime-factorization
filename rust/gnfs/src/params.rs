use serde::{Deserialize, Serialize};

/// Sieve algorithm selection.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SieveMode {
    Line,
    Lattice,
}

/// Full GNFS parameter set for a given digit size.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GnfsParams {
    pub name: String,
    pub degree: u32,
    pub lim0: u64,
    pub lim1: u64,
    pub lpb0: u32,
    pub lpb1: u32,
    pub mfb0: u32,
    pub mfb1: u32,
    pub sieve_a: u64,
    pub max_b: u64,
    pub rels_wanted: u64,
    pub qmin: u64,
    /// log2 of sieve half-width for lattice sieve (I = 2^log_i).
    pub log_i: u32,
    /// Sieve algorithm: Line (sweep all (a,b)) or Lattice (special-q).
    pub sieve_mode: SieveMode,
}

impl GnfsParams {
    pub fn c60() -> Self {
        Self {
            name: "c60".into(),
            degree: 4,
            lim0: 50_000,
            lim1: 50_000,
            lpb0: 20,
            lpb1: 20,
            mfb0: 40,
            mfb1: 40,
            sieve_a: 200_000,
            max_b: 50_000,
            rels_wanted: 50_000,
            qmin: 60_000,
            log_i: 11,
            sieve_mode: SieveMode::Lattice,
        }
    }

    pub fn c80() -> Self {
        Self {
            name: "c80".into(),
            degree: 5,
            lim0: 200_000,
            lim1: 200_000,
            lpb0: 23,
            lpb1: 23,
            mfb0: 46,
            mfb1: 46,
            sieve_a: 1_000_000,
            max_b: 100_000,
            rels_wanted: 500_000,
            qmin: 250_000,
            log_i: 13,
            sieve_mode: SieveMode::Lattice,
        }
    }

    pub fn test_small() -> Self {
        Self {
            name: "test_small".into(),
            degree: 3,
            lim0: 2_000,
            lim1: 2_000,
            lpb0: 16,
            lpb1: 16,
            mfb0: 32,
            mfb1: 32,
            sieve_a: 10_000,
            max_b: 2_000,
            rels_wanted: 2_000,
            qmin: 1_000,
            log_i: 8,
            sieve_mode: SieveMode::Line,
        }
    }

    /// Parameters for ~30-50 digit numbers (100-166 bits), matching CADO-NFS c30.
    pub fn c30() -> Self {
        Self {
            name: "c30".into(),
            degree: 3,
            lim0: 30_000,
            lim1: 30_000,
            lpb0: 17,
            lpb1: 17,
            mfb0: 18,
            mfb1: 18,
            sieve_a: 100_000,
            max_b: 5_000,
            rels_wanted: 30_000,
            qmin: 50_000,
            log_i: 9,
            sieve_mode: SieveMode::Lattice,
        }
    }

    /// Parameters for ~10-20 digit numbers (33-66 bits).
    pub fn c20() -> Self {
        Self {
            name: "c20".into(),
            degree: 3,
            lim0: 5_000,
            lim1: 5_000,
            lpb0: 17,
            lpb1: 17,
            mfb0: 34,
            mfb1: 34,
            sieve_a: 50_000,
            max_b: 10_000,
            rels_wanted: 10_000,
            qmin: 1_000,
            log_i: 8,
            sieve_mode: SieveMode::Line,
        }
    }

    pub fn for_bits(bits: u64) -> Self {
        if bits <= 40 {
            Self::test_small()
        } else if bits <= 70 {
            Self::c20()
        } else if bits <= 100 {
            Self::c30()
        } else if bits <= 200 {
            Self::c60()
        } else if bits <= 270 {
            Self::c80()
        } else {
            Self::c80()
        }
    }

    pub fn large_prime_bound_0(&self) -> u64 {
        1u64 << self.lpb0
    }

    pub fn large_prime_bound_1(&self) -> u64 {
        1u64 << self.lpb1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_params_presets() {
        let p = GnfsParams::c60();
        assert_eq!(p.degree, 4);
        assert_eq!(p.lim0, 50_000);
        assert!(p.large_prime_bound_0() == 1 << 20);
    }

    #[test]
    fn test_params_for_bits() {
        let small = GnfsParams::for_bits(30);
        assert_eq!(small.name, "test_small");
        let c20 = GnfsParams::for_bits(50);
        assert_eq!(c20.name, "c20");
        let c30 = GnfsParams::for_bits(80);
        assert_eq!(c30.name, "c30");
        let c60 = GnfsParams::for_bits(180);
        assert_eq!(c60.name, "c60");
    }
}
