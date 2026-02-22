//! # cf-factor
//!
//! Integer factoring via continued fractions and quadratic forms.
//!
//! Implements the Murru-Salvatori approach and SQUFOF theory for factoring integers
//! using continued fraction expansion of sqrt(N), reduced binary quadratic forms,
//! infrastructure distance in real quadratic fields, and Gauss composition.
//!
//! ## Algorithms
//!
//! - **SQUFOF** (Square Form Factorization): Shanks' algorithm, fastest for moderate N
//! - **Continued Fractions**: Expansion of sqrt(N) with convergent tracking
//! - **Gauss Composition**: Composition of binary quadratic forms for class group navigation
//! - **Regulator Estimation**: Period detection in CF expansion

pub mod cf;
pub mod factor;
pub mod forms;
pub mod regulator;
pub mod squfof;
