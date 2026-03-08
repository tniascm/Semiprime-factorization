//! GPU-Accelerated Block Wiedemann over GF(2)
//!
//! This crate implements the core sequence generation phase of the Block Wiedemann
//! algorithm for finding the nullspace of massive, sparse matrices over GF(2).
//! It is designed specifically to max out memory bandwidth and parallel compute
//! on Apple Silicon (M-series) via the Metal API.

#[cfg(target_os = "macos")]
pub mod gpu;

pub mod cpu;
pub mod matrix;

#[derive(Debug, thiserror::Error)]
pub enum WiedemannError {
    #[error("Metal initialization failed: {0}")]
    MetalError(String),
    #[error("Invalid matrix dimensions: {0}")]
    DimensionMismatch(String),
}
