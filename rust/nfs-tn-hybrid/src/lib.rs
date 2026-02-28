//! # NFS-TN Hybrid: Tensor Network optimization of NFS norm minimization
//!
//! Encodes the Number Field Sieve's joint norm minimization problem as a
//! QUBO (Quadratic Unconstrained Binary Optimization) Hamiltonian, then
//! uses Tree Tensor Network (TTN) variational optimization to find smooth
//! (a, b) pairs.
//!
//! ## Key Insight
//!
//! Standard TNSS (Tesoro/Siloi) encodes Schnorr's lattice CVP over integers
//! of size ~N, requiring n ~ l^5.8 binary variables. The NFS works with norms
//! of size N^{1/(d+1)}, dramatically improving smoothness probability. By
//! encoding NFS norm minimization as the TN cost function, we get:
//! - Far fewer binary variables: ~l/2 vs l^5.8
//! - NFS-quality smoothness probabilities
//! - Joint optimization of rational and algebraic norms
//!
//! ## Modules
//!
//! - [`hamiltonian`] - QUBO construction from NFS norms
//! - [`pipeline`] - End-to-end NFS-TN pipeline

pub mod hamiltonian;
pub mod pipeline;

/// Configuration for the NFS-TN hybrid.
#[derive(Debug, Clone)]
pub struct NfsTnConfig {
    /// Number of bits for the 'a' variable encoding.
    pub a_bits: usize,
    /// Number of bits for the 'b' variable encoding.
    pub b_bits: usize,
    /// NFS polynomial degree (3 for small, 4-5 for larger).
    pub poly_degree: usize,
    /// Weight for rational norm term in joint Hamiltonian.
    pub alpha_rational: f64,
    /// Weight for algebraic norm term in joint Hamiltonian.
    pub alpha_algebraic: f64,
    /// Penalty weight for coprimality constraint.
    pub coprimality_penalty: f64,
    /// TTN bond dimension (0 = auto from paper formula).
    pub bond_dim: usize,
    /// Number of TTN optimization sweeps.
    pub num_sweeps: usize,
    /// Number of samples to draw from optimized TTN.
    pub num_samples: usize,
    /// Factor base bound for smoothness checking.
    pub factor_base_bound: u64,
    /// Smoothness bound for the rational side.
    pub rational_smooth_bound: u64,
    /// Smoothness bound for the algebraic side.
    pub algebraic_smooth_bound: u64,
    /// Random seed.
    pub seed: u64,
}

impl NfsTnConfig {
    /// Create a config appropriate for the given bit size of N.
    pub fn for_bits(bits: u32) -> Self {
        // NFS polynomial degree
        let poly_degree = if bits <= 90 { 3 } else if bits <= 120 { 4 } else { 5 };

        // a,b encoding: sieve region size ~ N^{1/(d+1)}
        // a needs ~bits/(d+1) bits, b needs ~log2(max_b) bits
        let norm_bits = (bits as usize) / (poly_degree + 1);
        let a_bits = norm_bits + 2; // extra bits for sign offset
        let b_bits = norm_bits.max(4); // b > 0, at least 4 bits

        // Factor base bound: heuristic L[1/3] scaling
        let fb_bound = if bits <= 32 {
            100u64
        } else if bits <= 48 {
            500
        } else if bits <= 64 {
            2000
        } else if bits <= 80 {
            5000
        } else {
            10000
        };

        Self {
            a_bits,
            b_bits,
            poly_degree,
            alpha_rational: 1.0,
            alpha_algebraic: 1.0,
            coprimality_penalty: 10.0,
            bond_dim: 0, // auto
            num_sweeps: 20,
            num_samples: 500,
            factor_base_bound: fb_bound,
            rational_smooth_bound: fb_bound,
            algebraic_smooth_bound: fb_bound,
            seed: 42,
        }
    }

    /// Total number of primary binary variables (before quadratization auxiliaries).
    pub fn primary_vars(&self) -> usize {
        self.a_bits + self.b_bits
    }
}
