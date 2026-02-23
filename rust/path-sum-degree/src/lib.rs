/// E20: Boolean Polynomial Degree Audit for Eisenstein Channels
///
/// Tests the degree-rank equivalence hypothesis connecting Amy-Stinchcombe's
/// confluent path sum rewriting (bounded degree → poly-time classical simulation)
/// to the project's barrier theorem (bounded CRT rank → spectral flatness).
///
/// The core question: does the Boolean polynomial degree of
///   f_k,ℓ(N) = (σ_{k-1}(N) mod ℓ) mod 2
/// grow as O(1) (poly-time factoring via rewriting), O(n^c) for c < 1/3
/// (better-than-GNFS), or Ω(n) (barrier confirmed for poly-depth circuits)?

pub mod degree;
pub mod scaling;
