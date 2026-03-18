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
    /// Sieve threshold mfb values (original, before 2LP bump).
    /// Used for survivor detection to avoid inflating false positives.
    pub sieve_mfb0: u32,
    pub sieve_mfb1: u32,
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
            lim0: 20_000,
            lim1: 20_000,
            lpb0: 18,
            lpb1: 18,
            mfb0: 20,
            mfb1: 20,
            sieve_mfb0: 18,
            sieve_mfb1: 18,
            log_i: 8,
            qmin: 30_000,
            qrange: 500,
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
            sieve_mfb0: 20,
            sieve_mfb1: 20,
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
            sieve_mfb0: 22,
            sieve_mfb1: 22,
            log_i: 9,
            qmin: 35_000,
            qrange: 5_000,
            rels_wanted: 40_000,
        }
    }

    /// Parameters for ~45-digit (141+ bit) semiprimes.
    ///
    /// Tuned via parameter sweep (2026-03-13, updated 2026-03-16):
    /// - Smaller FB (lim 40k/45k vs 55k/65k) cuts dense columns, improving
    ///   row/col ratio and reducing matrix size.
    /// - Higher lpb (21/22 vs 20/21) yields 35% more relations per sieve second
    ///   and 7.5% faster total time; 2LP mfb is auto-bumped by pipeline.
    /// - Lower qmin (35k vs 58k) provides more diverse special-qs, improving
    ///   per-SQ yield and overall sieve throughput.
    /// - Smaller log_i (9 vs 10) halves the sieve region, roughly doubling
    ///   the number of special-qs processed in the same wall time.
    /// Net effect: 16.1s mean c45 total (3/3 factored, seed 42).
    pub fn c45() -> Self {
        Self {
            name: "c45",
            degree: 4,
            lim0: 40_000,
            lim1: 45_000,
            lpb0: 21,
            lpb1: 22,
            mfb0: 28,
            mfb1: 30,
            sieve_mfb0: 28,
            sieve_mfb1: 30,
            log_i: 9,
            qmin: 35_000,
            qrange: 750,
            rels_wanted: 45_000,
        }
    }

    /// c45 parameters for bucket sieve with larger sieve area (I=11).
    ///
    /// With log_i=11: sieve_width=4096, max_j=2048, area=8M (16x larger than I=9).
    /// Yields ~40 rels/SQ (vs 9 at I=9), needing ~850 SQs (vs 3835).
    /// Used automatically when bucket sieve is selected for degree >= 4.
    pub fn c45_bucket() -> Self {
        Self {
            name: "c45_bucket",
            degree: 4,
            lim0: 40_000,
            lim1: 45_000,
            lpb0: 20,
            lpb1: 21,
            mfb0: 28,
            mfb1: 30,
            sieve_mfb0: 28,
            sieve_mfb1: 30,
            log_i: 11,
            qmin: 35_000,
            qrange: 3_000,
            rels_wanted: 45_000,
        }
    }

    /// Parameters for ~60-digit (171+ bit) semiprimes.
    ///
    /// Smaller FB than CADO c60 reference to ensure relation surplus:
    /// - lim=30K/40K: ~11K dense columns vs ~17K expected relations → 6K surplus.
    ///   CADO uses lim=79K/111K but has 2-5x higher per-SQ yield from optimized
    ///   sieve + ECM cofactoring. Our lower yield requires smaller matrix.
    /// - lpb0=19, lpb1=20: slightly higher than CADO (18/19) to increase rels/SQ
    ///   with smaller FB. Matches CADO c65 lpb values.
    /// - mfb0=19 ≤ lpb0: no 2LP on rational side. mfb1=40=2×lpb1: full 2LP alg.
    /// - log_i=11: 16M sieve area. Sufficient with smaller FB.
    /// - qmin=30K within lim1=40K: all SQs fold into algebraic FB.
    pub fn c60() -> Self {
        Self {
            name: "c60",
            degree: 4,
            lim0: 80_000,   // CADO-exact for maximum filter yield (65%)
            lim1: 110_000,
            lpb0: 18,
            lpb1: 19,
            mfb0: 17,
            mfb1: 38,
            sieve_mfb0: 17,
            sieve_mfb1: 38,
            log_i: 9,
            qmin: 60_000,
            qrange: 50_000,
            rels_wanted: 30_000,
        }
    }

    /// Select parameters automatically based on semiprime bit-size.
    pub fn for_bits(bits: u32) -> Self {
        match bits {
            0..=105 => Self::c30(),
            106..=120 => Self::c35(),
            121..=140 => Self::c40(),
            141..=170 => Self::c45(),
            _ => Self::c60(),
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
        assert_eq!(p.large_prime_bound_0(), 262_144);
        assert_eq!(p.sieve_half_width(), 256);
    }

    #[test]
    fn test_c30_matches_tuned() {
        let p = NfsParams::c30();
        assert_eq!(p.log_i, 8);
        assert_eq!(p.lim0, 20_000);
        assert_eq!(p.lpb0, 18);
        assert_eq!(p.mfb0, 20);
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

    #[test]
    fn test_params_for_bits_c45_range() {
        let p148 = NfsParams::for_bits(148);
        assert_eq!(p148.name, "c45");

        let p170 = NfsParams::for_bits(170);
        assert_eq!(p170.name, "c45");
    }

    #[test]
    fn test_params_for_bits_c60() {
        let p171 = NfsParams::for_bits(171);
        assert_eq!(p171.name, "c60");

        let p199 = NfsParams::for_bits(199);
        assert_eq!(p199.name, "c60");
    }

    #[test]
    fn test_params_c60() {
        let p = NfsParams::c60();
        assert_eq!(p.degree, 4);
        assert_eq!(p.log_i, 9);
        assert_eq!(p.lim0, 80_000);
        assert_eq!(p.lim1, 110_000);
        assert_eq!(p.lpb0, 18);
        assert_eq!(p.lpb1, 19);
        assert_eq!(p.mfb0, 17);
        assert_eq!(p.mfb1, 38);
        assert_eq!(p.large_prime_bound_0(), 1 << 18);
        assert_eq!(p.large_prime_bound_1(), 1 << 19);
        assert_eq!(p.sieve_half_width(), 512);
    }
}
