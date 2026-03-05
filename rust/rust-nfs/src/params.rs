use serde::{Deserialize, Serialize};

/// CADO-NFS-matched parameter set for a given digit range.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NfsParams {
    pub name: &'static str,
    pub degree: u32,
    pub lim0: u64,
    pub lim1: u64,
    pub lpb0: u32,
    pub lpb1: u32,
    pub mfb0: u32,
    pub mfb1: u32,
    pub log_i: u32,
    pub qmin: u64,
    pub qrange: u64,
    pub rels_wanted: u64,
}

impl NfsParams {
    /// Parameters for ~30-digit (up to 105-bit) semiprimes.
    pub fn c30() -> Self {
        Self {
            name: "c30",
            degree: 3,
            lim0: 30_000,
            lim1: 30_000,
            lpb0: 17,
            lpb1: 17,
            mfb0: 18,
            mfb1: 18,
            log_i: 9,
            qmin: 50_000,
            qrange: 1_000,
            rels_wanted: 30_000,
        }
    }

    /// Parameters for ~35-digit (106-120 bit) semiprimes.
    pub fn c35() -> Self {
        Self {
            name: "c35",
            degree: 3,
            lim0: 40_000,
            lim1: 40_000,
            lpb0: 18,
            lpb1: 18,
            mfb0: 20,
            mfb1: 20,
            log_i: 9,
            qmin: 25_000,
            qrange: 5_000,
            rels_wanted: 35_000,
        }
    }

    /// Parameters for ~40-digit (121-140 bit) semiprimes.
    pub fn c40() -> Self {
        Self {
            name: "c40",
            degree: 4,
            lim0: 50_000,
            lim1: 55_000,
            lpb0: 18,
            lpb1: 18,
            mfb0: 22,
            mfb1: 22,
            log_i: 9,
            qmin: 35_000,
            qrange: 5_000,
            rels_wanted: 40_000,
        }
    }

    /// Parameters for ~45-digit (141+ bit) semiprimes.
    pub fn c45() -> Self {
        Self {
            name: "c45",
            degree: 4,
            lim0: 55_000,
            lim1: 65_000,
            lpb0: 18,
            lpb1: 19,
            mfb0: 24,
            mfb1: 26,
            log_i: 10,
            qmin: 58_000,
            qrange: 1_500,
            rels_wanted: 45_000,
        }
    }

    /// Select parameters automatically based on semiprime bit-size.
    pub fn for_bits(bits: u32) -> Self {
        match bits {
            0..=105 => Self::c30(),
            106..=120 => Self::c35(),
            121..=140 => Self::c40(),
            _ => Self::c45(),
        }
    }

    /// Rational large-prime bound: 2^lpb0.
    pub fn large_prime_bound_0(&self) -> u64 {
        1u64 << self.lpb0
    }

    /// Algebraic large-prime bound: 2^lpb1.
    pub fn large_prime_bound_1(&self) -> u64 {
        1u64 << self.lpb1
    }

    /// Half-width of the sieve region: 2^log_i.
    pub fn sieve_half_width(&self) -> u64 {
        1u64 << self.log_i
    }

    /// Full width of the sieve region: 2^(log_i + 1).
    pub fn sieve_width(&self) -> u64 {
        1u64 << (self.log_i + 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_params_c30() {
        let p = NfsParams::c30();
        assert_eq!(p.degree, 3);
        assert_eq!(p.large_prime_bound_0(), 131_072);
        assert_eq!(p.sieve_half_width(), 512);
    }

    #[test]
    fn test_params_for_bits() {
        let p96 = NfsParams::for_bits(96);
        assert_eq!(p96.name, "c30");

        let p112 = NfsParams::for_bits(112);
        assert_eq!(p112.name, "c35");

        let p128 = NfsParams::for_bits(128);
        assert_eq!(p128.name, "c40");
    }
}
