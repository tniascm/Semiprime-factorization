//! GNFS parameter space definition and genetic operators.
//!
//! Defines the tunable CADO-NFS parameters as a structured type with:
//! - Data-driven default parameters derived from CADO-NFS expert tables
//! - Search ranges computed by interpolating and expanding expert values
//! - Random initialization within valid ranges
//! - Mutation (tweak 1-3 parameters by +/-10-50%)
//! - Crossover (uniform blend of two configurations)
//! - Serialization to CADO-NFS CLI arguments

use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

// ---------------------------------------------------------------------------
// Expert parameter table — extracted from CADO-NFS params.c30 through params.c200
// ---------------------------------------------------------------------------

/// A single entry from the CADO-NFS expert parameter table.
/// Each corresponds to an official `params.cN` file.
#[derive(Debug, Clone, Copy)]
struct ExpertParams {
    /// Number of decimal digits (e.g., 30 for c30).
    digits: usize,
    /// Polynomial selection degree.
    degree: u32,
    /// Maximum leading coefficient for polynomial search.
    admax: u64,
    /// Rational factor base bound (tasks.lim0).
    lim0: u64,
    /// Algebraic factor base bound (tasks.lim1).
    lim1: u64,
    /// Rational large prime bits (tasks.lpb0).
    lpb0: u32,
    /// Algebraic large prime bits (tasks.lpb1).
    lpb1: u32,
    /// Rational cofactor bound bits (tasks.sieve.mfb0).
    mfb0: u32,
    /// Algebraic cofactor bound bits (tasks.sieve.mfb1).
    mfb1: u32,
    /// Special-q range per workunit.
    qrange: u64,
}

/// Expert parameter table extracted from official CADO-NFS parameter files.
/// Sorted by digit count. Values come directly from params.c30 through params.c200.
const EXPERT_PARAMS: &[ExpertParams] = &[
    ExpertParams { digits: 30,  degree: 3, admax: 5_000,      lim0: 30_000,      lim1: 30_000,      lpb0: 17, lpb1: 17, mfb0: 18, mfb1: 18, qrange: 1_000 },
    ExpertParams { digits: 35,  degree: 3, admax: 6_000,      lim0: 40_000,      lim1: 40_000,      lpb0: 18, lpb1: 18, mfb0: 20, mfb1: 20, qrange: 1_200 },
    ExpertParams { digits: 40,  degree: 3, admax: 7_000,      lim0: 50_000,      lim1: 55_000,      lpb0: 18, lpb1: 18, mfb0: 22, mfb1: 22, qrange: 1_500 },
    ExpertParams { digits: 45,  degree: 4, admax: 8_000,      lim0: 55_000,      lim1: 65_000,      lpb0: 18, lpb1: 19, mfb0: 24, mfb1: 26, qrange: 1_500 },
    ExpertParams { digits: 50,  degree: 4, admax: 8_500,      lim0: 65_000,      lim1: 80_000,      lpb0: 18, lpb1: 19, mfb0: 18, mfb1: 34, qrange: 1_500 },
    ExpertParams { digits: 55,  degree: 4, admax: 9_000,      lim0: 72_000,      lim1: 95_000,      lpb0: 18, lpb1: 19, mfb0: 17, mfb1: 36, qrange: 2_000 },
    ExpertParams { digits: 60,  degree: 4, admax: 10_000,     lim0: 78_682,      lim1: 111_342,     lpb0: 18, lpb1: 19, mfb0: 17, mfb1: 38, qrange: 2_000 },
    ExpertParams { digits: 65,  degree: 4, admax: 22_000,     lim0: 280_682,     lim1: 230_638,     lpb0: 19, lpb1: 20, mfb0: 18, mfb1: 40, qrange: 1_000 },
    ExpertParams { digits: 70,  degree: 4, admax: 44_000,     lim0: 343_245,     lim1: 244_248,     lpb0: 20, lpb1: 21, mfb0: 19, mfb1: 42, qrange: 1_000 },
    ExpertParams { digits: 75,  degree: 4, admax: 84_000,     lim0: 192_139,     lim1: 290_492,     lpb0: 21, lpb1: 21, mfb0: 41, mfb1: 42, qrange: 1_000 },
    ExpertParams { digits: 80,  degree: 4, admax: 100_000,    lim0: 292_877,     lim1: 339_976,     lpb0: 21, lpb1: 21, mfb0: 41, mfb1: 42, qrange: 5_000 },
    ExpertParams { digits: 85,  degree: 4, admax: 50_000,     lim0: 393_010,     lim1: 551_399,     lpb0: 22, lpb1: 22, mfb0: 44, mfb1: 44, qrange: 1_000 },
    ExpertParams { digits: 90,  degree: 4, admax: 100_000,    lim0: 404_327,     lim1: 811_066,     lpb0: 23, lpb1: 23, mfb0: 46, mfb1: 46, qrange: 10_000 },
    ExpertParams { digits: 95,  degree: 4, admax: 288,        lim0: 450_000,     lim1: 550_000,     lpb0: 24, lpb1: 25, mfb0: 47, mfb1: 48, qrange: 5_000 },
    ExpertParams { digits: 100, degree: 5, admax: 1_680,      lim0: 650_000,     lim1: 800_000,     lpb0: 25, lpb1: 26, mfb0: 48, mfb1: 51, qrange: 5_000 },
    ExpertParams { digits: 110, degree: 5, admax: 6_000,      lim0: 1_300_000,   lim1: 2_000_000,   lpb0: 26, lpb1: 27, mfb0: 50, mfb1: 52, qrange: 5_000 },
    ExpertParams { digits: 120, degree: 5, admax: 4_140,      lim0: 2_500_000,   lim1: 3_400_000,   lpb0: 27, lpb1: 28, mfb0: 52, mfb1: 54, qrange: 10_000 },
    ExpertParams { digits: 130, degree: 5, admax: 38_000,     lim0: 6_000_000,   lim1: 7_500_000,   lpb0: 28, lpb1: 29, mfb0: 54, mfb1: 57, qrange: 10_000 },
    ExpertParams { digits: 140, degree: 5, admax: 70_000,     lim0: 11_000_000,  lim1: 14_000_000,  lpb0: 30, lpb1: 30, mfb0: 57, mfb1: 58, qrange: 10_000 },
    ExpertParams { digits: 150, degree: 5, admax: 140_000,    lim0: 28_000_000,  lim1: 36_000_000,  lpb0: 31, lpb1: 31, mfb0: 59, mfb1: 60, qrange: 20_000 },
    ExpertParams { digits: 160, degree: 5, admax: 300_000,    lim0: 38_000_000,  lim1: 50_000_000,  lpb0: 31, lpb1: 32, mfb0: 60, mfb1: 61, qrange: 20_000 },
    ExpertParams { digits: 180, degree: 5, admax: 7_000_000,  lim0: 167_874_892, lim1: 91_296_860,  lpb0: 31, lpb1: 32, mfb0: 62, mfb1: 99, qrange: 5_000 },
    ExpertParams { digits: 200, degree: 5, admax: 10_000_000, lim0: 130_000_000, lim1: 100_000_000, lpb0: 32, lpb1: 33, mfb0: 85, mfb1: 96, qrange: 10_000 },
];

// ---------------------------------------------------------------------------
// Interpolation utilities
// ---------------------------------------------------------------------------

/// Convert a bit count to an approximate decimal digit count.
/// Uses ceil to be conservative (slightly larger params are safer than too-small).
fn bits_to_digits(n_bits: u32) -> usize {
    ((n_bits as f64) * std::f64::consts::LOG10_2).ceil() as usize
}

/// Linear interpolation for u32 values, rounding to nearest.
fn lerp_u32(a: u32, b: u32, t: f64) -> u32 {
    (a as f64 + (b as f64 - a as f64) * t).round() as u32
}

/// Log-linear interpolation for u64 values.
/// Interpolates in log-space, appropriate for parameters spanning orders of magnitude.
fn log_lerp_u64(a: u64, b: u64, t: f64) -> u64 {
    let log_a = (a.max(1) as f64).ln();
    let log_b = (b.max(1) as f64).ln();
    let log_val = log_a + (log_b - log_a) * t;
    log_val.exp().round() as u64
}

/// Find two expert entries that bracket the target digit count.
/// Returns (lower_index, upper_index) into EXPERT_PARAMS.
fn find_bracketing_indices(num_digits: usize) -> (usize, usize) {
    let table = EXPERT_PARAMS;

    // Below minimum
    if num_digits <= table[0].digits {
        return (0, 1.min(table.len() - 1));
    }

    // Above maximum
    if num_digits >= table[table.len() - 1].digits {
        return (table.len().saturating_sub(2), table.len() - 1);
    }

    // Find bracketing pair
    for i in 0..table.len() - 1 {
        if table[i].digits <= num_digits && num_digits <= table[i + 1].digits {
            return (i, i + 1);
        }
    }

    // Fallback (should never reach here)
    let mid = table.len() / 2;
    (mid, mid + 1)
}

/// Interpolate expert parameters for a given digit count.
///
/// Uses linear interpolation for integer params (degree, lpb, mfb)
/// and log-linear interpolation for magnitude params (admax, lim, qrange).
fn interpolate_expert(num_digits: usize) -> ExpertParams {
    let (lo_idx, hi_idx) = find_bracketing_indices(num_digits);
    let lo = &EXPERT_PARAMS[lo_idx];
    let hi = &EXPERT_PARAMS[hi_idx];

    if lo.digits == hi.digits || num_digits <= lo.digits {
        return *lo;
    }
    if num_digits >= hi.digits {
        return *hi;
    }

    let t = (num_digits - lo.digits) as f64 / (hi.digits - lo.digits) as f64;

    ExpertParams {
        digits: num_digits,
        degree: lerp_u32(lo.degree, hi.degree, t),
        admax: log_lerp_u64(lo.admax, hi.admax, t),
        lim0: log_lerp_u64(lo.lim0, hi.lim0, t),
        lim1: log_lerp_u64(lo.lim1, hi.lim1, t),
        lpb0: lerp_u32(lo.lpb0, hi.lpb0, t),
        lpb1: lerp_u32(lo.lpb1, hi.lpb1, t),
        mfb0: lerp_u32(lo.mfb0, hi.mfb0, t),
        mfb1: lerp_u32(lo.mfb1, hi.mfb1, t),
        qrange: log_lerp_u64(lo.qrange, hi.qrange, t),
    }
}

// ---------------------------------------------------------------------------
// Data-driven parameter ranges
// ---------------------------------------------------------------------------

/// Compute valid parameter ranges for a given digit count.
///
/// Looks at expert entries in a window around the target, computes min/max,
/// then expands by a factor to allow evolutionary exploration beyond
/// expert-tuned values.
fn ranges_for_digits(num_digits: usize) -> ParamRanges {
    // Window: look at entries within +/-15 digits of the target.
    let window = 15;
    let lo_digits = num_digits.saturating_sub(window);
    let hi_digits = num_digits + window;

    let entries: Vec<&ExpertParams> = EXPERT_PARAMS
        .iter()
        .filter(|e| e.digits >= lo_digits && e.digits <= hi_digits)
        .collect();

    // If window is too narrow (very small or very large), use nearest 3
    let entries = if entries.len() < 2 {
        let mut sorted: Vec<&ExpertParams> = EXPERT_PARAMS.iter().collect();
        sorted.sort_by_key(|e| (e.digits as i64 - num_digits as i64).unsigned_abs());
        sorted.into_iter().take(3).collect::<Vec<_>>()
    } else {
        entries
    };

    // Compute min/max across window for each parameter
    let degree_min = entries.iter().map(|e| e.degree).min().unwrap();
    let degree_max = entries.iter().map(|e| e.degree).max().unwrap();
    let admax_min = entries.iter().map(|e| e.admax).min().unwrap();
    let admax_max = entries.iter().map(|e| e.admax).max().unwrap();
    let lim0_min = entries.iter().map(|e| e.lim0).min().unwrap();
    let lim0_max = entries.iter().map(|e| e.lim0).max().unwrap();
    let lim1_min = entries.iter().map(|e| e.lim1).min().unwrap();
    let lim1_max = entries.iter().map(|e| e.lim1).max().unwrap();
    let lpb0_min = entries.iter().map(|e| e.lpb0).min().unwrap();
    let lpb0_max = entries.iter().map(|e| e.lpb0).max().unwrap();
    let lpb1_min = entries.iter().map(|e| e.lpb1).min().unwrap();
    let lpb1_max = entries.iter().map(|e| e.lpb1).max().unwrap();
    let mfb0_min = entries.iter().map(|e| e.mfb0).min().unwrap();
    let mfb0_max = entries.iter().map(|e| e.mfb0).max().unwrap();
    let mfb1_min = entries.iter().map(|e| e.mfb1).min().unwrap();
    let mfb1_max = entries.iter().map(|e| e.mfb1).max().unwrap();
    let qrange_min = entries.iter().map(|e| e.qrange).min().unwrap();
    let qrange_max = entries.iter().map(|e| e.qrange).max().unwrap();

    // Expand log-scale params by x/÷ 1.5
    let expand_log = |min_val: u64, max_val: u64| -> (u64, u64) {
        let expanded_min = ((min_val as f64) / 1.5).round() as u64;
        let expanded_max = ((max_val as f64) * 1.5).round() as u64;
        (expanded_min.max(1), expanded_max)
    };

    // Expand integer params by +/-2
    let expand_int = |min_val: u32, max_val: u32, by: u32| -> (u32, u32) {
        (min_val.saturating_sub(by), max_val + by)
    };

    let (admax_lo, admax_hi) = expand_log(admax_min, admax_max);
    let (lim0_lo, lim0_hi) = expand_log(lim0_min, lim0_max);
    let (lim1_lo, lim1_hi) = expand_log(lim1_min, lim1_max);
    let (qrange_lo, qrange_hi) = expand_log(qrange_min, qrange_max);
    let (lpb0_lo, lpb0_hi) = expand_int(lpb0_min, lpb0_max, 2);
    let (lpb1_lo, lpb1_hi) = expand_int(lpb1_min, lpb1_max, 2);
    let (mfb0_lo, mfb0_hi) = expand_int(mfb0_min, mfb0_max, 4);
    let (mfb1_lo, mfb1_hi) = expand_int(mfb1_min, mfb1_max, 4);

    // Compute incr max based on digit count
    let incr_max = if num_digits <= 50 { 120 } else if num_digits <= 100 { 240 } else { 420 };

    // Apply absolute floor/ceiling constraints
    ParamRanges {
        poly_degree: (degree_min.max(3), degree_max.min(6)),
        poly_admax: (admax_lo.max(100), admax_hi),
        poly_incr: (60, incr_max),
        fb_rational_bound: (lim0_lo.max(1_000), lim0_hi),
        fb_algebraic_bound: (lim1_lo.max(1_000), lim1_hi),
        lp_rational_bits: (lpb0_lo.max(15), lpb0_hi.min(35)),
        lp_algebraic_bits: (lpb1_lo.max(15), lpb1_hi.min(35)),
        sieve_mfbr: (mfb0_lo.max(15), mfb0_hi.min(100)),
        sieve_mfba: (mfb1_lo.max(15), mfb1_hi.min(100)),
        sieve_qrange: (qrange_lo.max(100), qrange_hi),
    }
}

// ---------------------------------------------------------------------------
// CadoParams and ParamRanges
// ---------------------------------------------------------------------------

/// Complete set of tunable CADO-NFS parameters for GNFS.
///
/// These are the ~10 most impactful parameters that affect GNFS performance.
/// Each maps directly to a CADO-NFS command-line flag.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CadoParams {
    /// Polynomial degree for NFS polynomial selection.
    /// CADO flag: `tasks.polyselect.degree`
    pub poly_degree: u32,

    /// Maximum leading coefficient for polynomial search.
    /// CADO flag: `tasks.polyselect.admax`
    pub poly_admax: u64,

    /// Increment step for leading coefficient search.
    /// CADO flag: `tasks.polyselect.incr`
    /// Must be divisible by 60.
    pub poly_incr: u64,

    /// Rational factor base bound.
    /// CADO flag: `tasks.lim0`
    pub fb_rational_bound: u64,

    /// Algebraic factor base bound.
    /// CADO flag: `tasks.lim1`
    pub fb_algebraic_bound: u64,

    /// Rational large prime bits.
    /// CADO flag: `tasks.lpb0`
    pub lp_rational_bits: u32,

    /// Algebraic large prime bits.
    /// CADO flag: `tasks.lpb1`
    pub lp_algebraic_bits: u32,

    /// Rational cofactor bound (bits).
    /// CADO flag: `tasks.sieve.mfb0`
    pub sieve_mfbr: u32,

    /// Algebraic cofactor bound (bits).
    /// CADO flag: `tasks.sieve.mfb1`
    pub sieve_mfba: u32,

    /// Special-q range per workunit.
    /// CADO flag: `tasks.sieve.qrange`
    pub sieve_qrange: u64,
}

/// Valid ranges for each parameter, used by random initialization and mutation.
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
    ///
    /// Derives ranges from the CADO-NFS expert parameter table by
    /// converting bits to digits and looking up the nearest entries.
    fn for_bits(n_bits: u32) -> Self {
        let num_digits = bits_to_digits(n_bits);
        ranges_for_digits(num_digits)
    }
}

impl CadoParams {
    /// CADO-NFS default parameters for a given bit size.
    ///
    /// Interpolated from the official CADO-NFS parameter files (params.c30
    /// through params.c200). Provides a good starting point that matches
    /// what CADO-NFS would use internally.
    pub fn default_for_bits(n_bits: u32) -> Self {
        let num_digits = bits_to_digits(n_bits);
        let expert = interpolate_expert(num_digits);

        CadoParams {
            poly_degree: expert.degree,
            poly_admax: expert.admax,
            poly_incr: 60, // CADO-NFS default; not in expert files
            fb_rational_bound: expert.lim0,
            fb_algebraic_bound: expert.lim1,
            lp_rational_bits: expert.lpb0,
            lp_algebraic_bits: expert.lpb1,
            // Expert files sometimes have mfb < lpb (intentional for CADO-NFS).
            // We enforce the minimum constraint mfb >= lpb.
            sieve_mfbr: expert.mfb0.max(expert.lpb0),
            sieve_mfba: expert.mfb1.max(expert.lpb1),
            sieve_qrange: expert.qrange,
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
    /// Each selected parameter is adjusted by +/-10-50% (multiplicative).
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
                    // poly_degree: +/-1
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

        // Enforce constraint: mfb >= lpb (matches CADO-NFS expert behavior)
        child.sieve_mfbr = child.sieve_mfbr.max(child.lp_rational_bits);
        child.sieve_mfba = child.sieve_mfba.max(child.lp_algebraic_bits);

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

        // Enforce constraint: mfb >= lpb (matches CADO-NFS expert behavior)
        child.sieve_mfbr = child.sieve_mfbr.max(child.lp_rational_bits);
        child.sieve_mfba = child.sieve_mfba.max(child.lp_algebraic_bits);

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
/// (e.g., factor base bounds from 1e4 to 1e8).
fn log_uniform(rng: &mut impl Rng, min: u64, max: u64) -> u64 {
    let log_min = (min.max(1) as f64).ln();
    let log_max = (max.max(1) as f64).ln();
    if log_max <= log_min {
        return min;
    }
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

    // --- Expert table and interpolation tests ---

    #[test]
    fn test_bits_to_digits() {
        assert_eq!(bits_to_digits(100), 31);  // 100 * 0.30103 = 30.103 -> 31
        assert_eq!(bits_to_digits(150), 46);  // 150 * 0.30103 = 45.155 -> 46
        assert_eq!(bits_to_digits(200), 61);  // 200 * 0.30103 = 60.206 -> 61
        assert_eq!(bits_to_digits(332), 100); // 332 * 0.30103 = 99.94 -> 100
    }

    #[test]
    fn test_expert_table_sorted() {
        for i in 1..EXPERT_PARAMS.len() {
            assert!(
                EXPERT_PARAMS[i].digits > EXPERT_PARAMS[i - 1].digits,
                "Expert table must be sorted: c{} after c{}",
                EXPERT_PARAMS[i].digits,
                EXPERT_PARAMS[i - 1].digits,
            );
        }
    }

    #[test]
    fn test_expert_table_degree_range() {
        for entry in EXPERT_PARAMS {
            assert!(entry.degree >= 3 && entry.degree <= 6,
                "c{}: degree {} out of range", entry.digits, entry.degree);
            assert!(entry.lpb0 >= 15 && entry.lpb0 <= 35,
                "c{}: lpb0 {} out of range", entry.digits, entry.lpb0);
            assert!(entry.lpb1 >= 15 && entry.lpb1 <= 35,
                "c{}: lpb1 {} out of range", entry.digits, entry.lpb1);
        }
    }

    #[test]
    fn test_interpolate_exact_match() {
        let expert = interpolate_expert(30);
        assert_eq!(expert.degree, 3);
        assert_eq!(expert.lim0, 30_000);
        assert_eq!(expert.lpb0, 17);
        assert_eq!(expert.mfb0, 18);
    }

    #[test]
    fn test_interpolate_between_entries() {
        // c32 should interpolate between c30 and c35
        let expert = interpolate_expert(32);
        assert_eq!(expert.degree, 3); // Both neighbors have degree 3
        assert!(expert.lim0 > 30_000 && expert.lim0 < 40_000,
            "c32 lim0={} should be between 30K and 40K", expert.lim0);
    }

    #[test]
    fn test_interpolate_below_minimum() {
        let expert = interpolate_expert(20);
        assert_eq!(expert.degree, 3);
        assert_eq!(expert.lim0, 30_000); // Clamps to c30
    }

    #[test]
    fn test_interpolate_above_maximum() {
        let expert = interpolate_expert(300);
        assert_eq!(expert.degree, 5); // Clamps to c200
    }

    #[test]
    fn test_ranges_contain_expert_values() {
        // Critical test: for each expert entry, verify the search ranges
        // at that digit count actually contain the expert's optimal values.
        for entry in EXPERT_PARAMS {
            let ranges = ranges_for_digits(entry.digits);

            assert!(
                entry.degree >= ranges.poly_degree.0 && entry.degree <= ranges.poly_degree.1,
                "c{}: expert degree={} outside range {:?}",
                entry.digits, entry.degree, ranges.poly_degree
            );
            assert!(
                entry.admax >= ranges.poly_admax.0 && entry.admax <= ranges.poly_admax.1,
                "c{}: expert admax={} outside range {:?}",
                entry.digits, entry.admax, ranges.poly_admax
            );
            assert!(
                entry.lim0 >= ranges.fb_rational_bound.0 && entry.lim0 <= ranges.fb_rational_bound.1,
                "c{}: expert lim0={} outside range {:?}",
                entry.digits, entry.lim0, ranges.fb_rational_bound
            );
            assert!(
                entry.lim1 >= ranges.fb_algebraic_bound.0 && entry.lim1 <= ranges.fb_algebraic_bound.1,
                "c{}: expert lim1={} outside range {:?}",
                entry.digits, entry.lim1, ranges.fb_algebraic_bound
            );
            assert!(
                entry.lpb0 >= ranges.lp_rational_bits.0 && entry.lpb0 <= ranges.lp_rational_bits.1,
                "c{}: expert lpb0={} outside range {:?}",
                entry.digits, entry.lpb0, ranges.lp_rational_bits
            );
            assert!(
                entry.lpb1 >= ranges.lp_algebraic_bits.0 && entry.lpb1 <= ranges.lp_algebraic_bits.1,
                "c{}: expert lpb1={} outside range {:?}",
                entry.digits, entry.lpb1, ranges.lp_algebraic_bits
            );
            assert!(
                entry.mfb0 >= ranges.sieve_mfbr.0 && entry.mfb0 <= ranges.sieve_mfbr.1,
                "c{}: expert mfb0={} outside range {:?}",
                entry.digits, entry.mfb0, ranges.sieve_mfbr
            );
            assert!(
                entry.mfb1 >= ranges.sieve_mfba.0 && entry.mfb1 <= ranges.sieve_mfba.1,
                "c{}: expert mfb1={} outside range {:?}",
                entry.digits, entry.mfb1, ranges.sieve_mfba
            );
            assert!(
                entry.qrange >= ranges.sieve_qrange.0 && entry.qrange <= ranges.sieve_qrange.1,
                "c{}: expert qrange={} outside range {:?}",
                entry.digits, entry.qrange, ranges.sieve_qrange
            );
        }
    }

    // --- Existing tests (updated assertions) ---

    #[test]
    fn test_default_params_100bit() {
        // 100 bits ~ 31 digits ~ c30/c35 interpolation
        let params = CadoParams::default_for_bits(100);
        assert_eq!(params.poly_degree, 3);
        assert!(params.fb_rational_bound > 0);
        assert!(params.lp_rational_bits >= 15);
        assert!(params.sieve_mfbr >= params.lp_rational_bits);
    }

    #[test]
    fn test_default_params_150bit() {
        // 150 bits ~ 46 digits ~ c45/c50 interpolation
        let params = CadoParams::default_for_bits(150);
        assert!(params.poly_degree >= 4);
        assert!(params.fb_rational_bound > 10_000);
    }

    #[test]
    fn test_default_params_200bit() {
        // 200 bits ~ 61 digits ~ c60/c65 interpolation
        let params = CadoParams::default_for_bits(200);
        assert!(params.poly_degree >= 4);
    }

    #[test]
    fn test_random_params() {
        let mut rng = thread_rng();
        for bits in [80, 100, 120, 150, 180, 200] {
            let params = CadoParams::random(&mut rng, bits);
            assert!(params.poly_degree >= 3 && params.poly_degree <= 6);
            assert!(params.fb_rational_bound > 0);
            assert!(params.lp_rational_bits >= 15 && params.lp_rational_bits <= 35);
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
            assert!(mutated.lp_rational_bits >= 15 && mutated.lp_rational_bits <= 35);
            assert!(mutated.sieve_mfbr >= mutated.lp_rational_bits);
            assert!(mutated.sieve_mfba >= mutated.lp_algebraic_bits);
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
        assert!(child.sieve_mfbr >= child.lp_rational_bits);
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
