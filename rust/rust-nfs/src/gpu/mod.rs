//! GPU Acceleration for Pillar 2: Hardware Dominance
//!
//! This module acts as the entry point for GPU kernels via Metal (or CUDA/OpenCL).
//! Below, the `include_str!` macros bake the `.metal` shader files into the binary.
//! The implementations load and use these strings if running on an Apple Silicon
//! machine where `metal` API calls are possible, otherwise they fall back gracefully.

pub mod ecm;
pub mod matrix;
pub mod sieve;

/// Source code for the bucket sieving kernel.
pub const SIEVE_SHADER_SRC: &str = include_str!("shaders/sieve.metal");

/// Source code for the GF(2) SpMV matrix multiplication kernel.
pub const MATRIX_SHADER_SRC: &str = include_str!("shaders/matrix.metal");

/// Source code for the ECM batched cofactorization kernel.
pub const ECM_SHADER_SRC: &str = include_str!("shaders/ecm.metal");
