//! cado-evolve: Evolutionary GNFS parameter optimization via CADO-NFS.
//!
//! Wraps CADO-NFS (production L[1/3] GNFS implementation) as an evaluation
//! backend, with an island-model evolutionary search tuning parameter
//! configurations for optimal factoring performance.
//!
//! Each "individual" in evolution = a complete CADO-NFS parameter configuration.
//! Target range: 100–200 bit semiprimes.

pub mod analysis;
pub mod benchmark;
pub mod cado;
pub mod evolution;
pub mod fitness;
pub mod params;
