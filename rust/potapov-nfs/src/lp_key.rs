use serde::{Deserialize, Serialize};

/// Side-aware large-prime ideal key used by partial-relation merging.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum LpKey {
    /// Rational-side large prime.
    Rational(u64),
    /// Algebraic-side large-prime ideal `(p, r)`.
    /// Projective ideals use `(p, p)`.
    Algebraic(u64, u64),
}
