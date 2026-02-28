//! E29: Novel Relation Collection via Joint Norm Biasing
//!
//! Tests whether MCMC sampling biased toward (a,b) pairs with smaller
//! log|a-bm| + log|F(a,b)| yields higher joint smoothness rates than
//! uniform random sampling within the same NFS sieve range.

pub mod logger;
pub mod sampler;
pub mod smoothness;

/// Experiment configuration.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct E29Config {
    pub bit_sizes: Vec<u32>,
    pub semiprimes_per_size: usize,
    pub candidates_per_method: usize,
    pub mcmc_chains: usize,
    pub mcmc_t_start: f64,
    pub mcmc_t_end: f64,
    pub seed: u64,
}

impl Default for E29Config {
    fn default() -> Self {
        Self {
            bit_sizes: vec![32, 40, 48, 56, 64],
            semiprimes_per_size: 20,
            candidates_per_method: 10_000,
            mcmc_chains: 10,
            mcmc_t_start: 10.0,
            mcmc_t_end: 0.1,
            seed: 42,
        }
    }
}

/// Sieve range parameters for a given bit size.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SieveParams {
    pub sieve_area: i64,
    pub max_b: i64,
    pub fb_bound: u64,
    pub degree: usize,
}

impl SieveParams {
    /// Get sieve parameters matching classical-nfs NfsPipelineParams::for_bits.
    pub fn for_bits(bits: u32) -> Self {
        if bits <= 40 {
            Self { sieve_area: 1 << 12, max_b: 1 << 8, fb_bound: 1 << 8, degree: 3 }
        } else if bits <= 50 {
            Self { sieve_area: 1 << 14, max_b: 1 << 9, fb_bound: 1 << 10, degree: 3 }
        } else if bits <= 60 {
            Self { sieve_area: 1 << 15, max_b: 1 << 10, fb_bound: 1 << 11, degree: 3 }
        } else if bits <= 70 {
            Self { sieve_area: 1 << 16, max_b: 1 << 11, fb_bound: 1 << 12, degree: 3 }
        } else {
            Self { sieve_area: 1 << 17, max_b: 1 << 12, fb_bound: 1 << 14, degree: 3 }
        }
    }
}

/// Result from one sampling method on one semiprime.
#[derive(Debug, Clone, serde::Serialize)]
pub struct MethodResult {
    pub method: String,
    pub candidates_tested: usize,
    pub valid_candidates: usize,
    pub rational_smooth: usize,
    pub algebraic_smooth: usize,
    pub both_smooth: usize,
    pub mean_rat_norm_log2: f64,
    pub mean_alg_norm_log2: f64,
    pub mean_energy: f64,
    pub time_ms: f64,
}

impl MethodResult {
    pub fn smooth_rate(&self) -> f64 {
        if self.valid_candidates == 0 {
            0.0
        } else {
            self.both_smooth as f64 / self.valid_candidates as f64
        }
    }
}

/// Per-semiprime comparison result.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ComparisonResult {
    pub n: u64,
    pub bits: u32,
    pub poly_degree: usize,
    pub m: u64,
    pub uniform: MethodResult,
    pub mcmc: MethodResult,
    pub smooth_rate_ratio: f64,
}
