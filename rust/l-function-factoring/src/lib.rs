//! # L-function Factoring
//!
//! A novel factoring method via Dirichlet L-functions and conductor detection.
//!
//! Uses the insight that characters mod N with conductor < N reveal divisors of N.
//! For N = pq, the Dirichlet characters mod N decompose as products via CRT,
//! and finding a non-trivial character with small conductor reveals the factorization.

pub mod characters;
pub mod class_number;
pub mod complex;
pub mod factor;
pub mod gauss_sums;
pub mod l_function;
pub mod sampling;
