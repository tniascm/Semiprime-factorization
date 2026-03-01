use serde::{Deserialize, Serialize};

/// A sieve relation: (a, b) pair with factorizations on both sides.
///
/// This extends `gnfs::types::Relation` with cofactor fields for
/// large-prime tracking (partial relations).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Relation {
    pub a: i64,
    pub b: u64,
    /// Rational-side factors as (prime_index, exponent) pairs.
    pub rational_factors: Vec<(u32, u8)>,
    /// Algebraic-side factors as (flat_index, exponent) pairs.
    pub algebraic_factors: Vec<(u32, u8)>,
    pub rational_sign_negative: bool,
    pub algebraic_sign_negative: bool,
    /// Rational cofactor: 0 or 1 if fully smooth, otherwise a large prime.
    pub rat_cofactor: u64,
    /// Algebraic cofactor: 0 or 1 if fully smooth, otherwise a large prime.
    pub alg_cofactor: u64,
}

impl Relation {
    /// A full relation has both cofactors <= 1 (fully factored).
    pub fn is_full(&self) -> bool {
        self.rat_cofactor <= 1 && self.alg_cofactor <= 1
    }

    /// A partial relation has at least one large-prime cofactor.
    pub fn is_partial(&self) -> bool {
        !self.is_full()
    }

    /// Convert to the `gnfs::types::Relation` used by linear algebra and
    /// square root stages. Only valid for full relations.
    pub fn to_gnfs_relation(&self) -> gnfs::types::Relation {
        gnfs::types::Relation {
            a: self.a,
            b: self.b,
            rational_factors: self.rational_factors.clone(),
            algebraic_factors: self.algebraic_factors.clone(),
            rational_sign_negative: self.rational_sign_negative,
            algebraic_sign_negative: self.algebraic_sign_negative,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_full() -> Relation {
        Relation {
            a: 17,
            b: 3,
            rational_factors: vec![(0, 1), (2, 2)],
            algebraic_factors: vec![(1, 1)],
            rational_sign_negative: false,
            algebraic_sign_negative: true,
            rat_cofactor: 1,
            alg_cofactor: 1,
        }
    }

    fn sample_partial() -> Relation {
        Relation {
            a: -5,
            b: 7,
            rational_factors: vec![(0, 1)],
            algebraic_factors: vec![(0, 2)],
            rational_sign_negative: true,
            algebraic_sign_negative: false,
            rat_cofactor: 65537,
            alg_cofactor: 1,
        }
    }

    #[test]
    fn test_is_full() {
        assert!(sample_full().is_full());
        assert!(!sample_full().is_partial());
    }

    #[test]
    fn test_is_partial() {
        assert!(sample_partial().is_partial());
        assert!(!sample_partial().is_full());
    }

    #[test]
    fn test_to_gnfs_relation() {
        let r = sample_full();
        let g = r.to_gnfs_relation();
        assert_eq!(g.a, 17);
        assert_eq!(g.b, 3);
        assert_eq!(g.rational_factors, vec![(0, 1), (2, 2)]);
        assert_eq!(g.algebraic_factors, vec![(1, 1)]);
        assert!(!g.rational_sign_negative);
        assert!(g.algebraic_sign_negative);
    }
}
