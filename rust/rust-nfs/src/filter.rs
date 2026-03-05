//! Filtering: dedup + sparse-column singleton removal.

use std::collections::{HashMap, HashSet};

use crate::lp_key::LpKey;
use crate::relation::Relation;

fn alg_lp_ideal_key(a: i64, b: u64, p: u64) -> Option<(u64, u64)> {
    if p < 2 {
        return None;
    }
    let b_mod_p = b % p;
    if b_mod_p == 0 {
        return Some((p, p));
    }
    let b_inv = match gnfs::arith::mod_inverse_u64(b_mod_p, p) {
        Some(v) => v,
        None => return Some((p, p)),
    };
    let a_mod_p = (a as i128).rem_euclid(p as i128) as u64;
    let r = ((a_mod_p as u128 * b_inv as u128) % p as u128) as u64;
    Some((p, r))
}

fn relation_lp_keys(rel: &Relation) -> Vec<LpKey> {
    if !rel.lp_keys.is_empty() {
        let mut parity = HashSet::new();
        for &k in &rel.lp_keys {
            if !parity.remove(&k) {
                parity.insert(k);
            }
        }
        let mut keys: Vec<LpKey> = parity.into_iter().collect();
        keys.sort_unstable();
        return keys;
    }

    // Backward-compatible fallback for relations that only populated legacy fields.
    let mut keys = Vec::new();
    if rel.rat_cofactor > 1 {
        keys.push(LpKey::Rational(rel.rat_cofactor));
    }
    if rel.alg_cofactor > 1 {
        if let Some((p, r)) = alg_lp_ideal_key(rel.a, rel.b, rel.alg_cofactor) {
            keys.push(LpKey::Algebraic(p, r));
        }
    }
    keys.sort_unstable();
    keys.dedup();
    keys
}

/// Remove exact duplicate relations and prune sparse singleton columns.
///
/// Strategy:
/// 1. Deduplicate by `(a, b, special_q)`.
/// 2. Iteratively remove relations containing singleton sparse columns:
///    - `special_q`
///    - LP ideal keys (`Relation.lp_keys`, with legacy cofactor fallback)
///
/// We intentionally do NOT run full dense-column singleton elimination here
/// (CADO does this after richer merge stages). Removing only sparse singletons
/// preserves matrix quality without collapsing relation count.
pub fn filter_relations(relations: Vec<Relation>) -> Vec<Relation> {
    // Step 1: Deduplicate by (a, b, special_q). The same (a,b) can appear
    // under different special-q ideals.
    let mut seen = HashSet::new();
    let mut filtered: Vec<Relation> = relations
        .into_iter()
        .filter(|r| seen.insert((r.a, r.b, r.special_q)))
        .collect();

    // Step 2: Iterative sparse singleton removal.
    loop {
        let mut sq_count: HashMap<(u64, u64), usize> = HashMap::new();
        let mut lp_count: HashMap<LpKey, usize> = HashMap::new();

        for rel in &filtered {
            if let Some(sq) = rel.special_q {
                *sq_count.entry(sq).or_insert(0) += 1;
            }
            for key in relation_lp_keys(rel) {
                *lp_count.entry(key).or_insert(0) += 1;
            }
        }

        let before = filtered.len();
        filtered.retain(|rel| {
            if let Some(sq) = rel.special_q {
                if sq_count.get(&sq).copied().unwrap_or(0) < 2 {
                    return false;
                }
            }
            for key in relation_lp_keys(rel) {
                if lp_count.get(&key).copied().unwrap_or(0) < 2 {
                    return false;
                }
            }
            true
        });

        if filtered.len() == before {
            break;
        }
    }

    filtered
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::relation::Relation;

    fn make_rel(a: i64, b: u64, rat: Vec<(u32, u8)>, alg: Vec<(u32, u8)>) -> Relation {
        Relation {
            a,
            b,
            rational_factors: rat,
            algebraic_factors: alg,
            rational_sign_negative: false,
            algebraic_sign_negative: false,
            special_q: None,
            rat_cofactor: 0,
            alg_cofactor: 0,
            lp_keys: vec![],
        }
    }

    #[test]
    fn test_dedup() {
        let rels = vec![
            make_rel(1, 2, vec![(0, 1)], vec![(0, 1)]),
            make_rel(1, 2, vec![(0, 1)], vec![(0, 1)]), // duplicate
            make_rel(3, 4, vec![(0, 1)], vec![(0, 1)]),
        ];
        let filtered = filter_relations(rels);
        assert_eq!(filtered.len(), 2);
    }

    #[test]
    fn test_sparse_singleton_removal() {
        let rels = vec![
            make_rel(1, 1, vec![(0, 1), (1, 1)], vec![(0, 1)]),
            make_rel(2, 1, vec![(0, 1)], vec![(0, 1)]),
            make_rel(3, 1, vec![(0, 1)], vec![(0, 1)]),
        ];
        let mut rels = rels;
        rels[0].rat_cofactor = 1009; // singleton LP => removed
        rels[1].rat_cofactor = 1013; // singleton LP => removed
        rels[2].rat_cofactor = 1013; // appears twice with rel[1]
        let filtered = filter_relations(rels);
        assert_eq!(filtered.len(), 2);
    }

    #[test]
    fn test_cascading_sparse_singleton_removal() {
        let rels = vec![
            make_rel(1, 1, vec![(0, 1), (1, 1)], vec![]),
            make_rel(2, 1, vec![(1, 1), (2, 1)], vec![]),
            make_rel(3, 1, vec![(2, 1), (3, 1)], vec![]),
            make_rel(4, 1, vec![(3, 1)], vec![]),
        ];
        let mut rels = rels;
        rels[0].special_q = Some((1009, 3)); // singleton
        rels[1].special_q = Some((1013, 5)); // appears twice initially
        rels[2].special_q = Some((1013, 5));
        rels[3].special_q = Some((1019, 7)); // singleton
        let filtered = filter_relations(rels);
        // singleton sq relations removed; (1013,5) survives
        assert_eq!(filtered.len(), 2);
    }

    #[test]
    fn test_dense_columns_not_pruned() {
        // No sparse columns set: dense columns are not singleton-pruned here.
        let rels = vec![
            make_rel(1, 1, vec![(0, 1), (1, 1)], vec![]),
            make_rel(2, 1, vec![(0, 1), (1, 1)], vec![]),
            make_rel(3, 1, vec![(0, 1)], vec![]),
        ];
        let filtered = filter_relations(rels);
        assert_eq!(filtered.len(), 3);
    }

    #[test]
    fn test_empty() {
        let filtered = filter_relations(vec![]);
        assert!(filtered.is_empty());
    }

    #[test]
    fn test_large_prime_singletons_removed() {
        let mut r1 = make_rel(1, 1, vec![(0, 1)], vec![]);
        r1.rat_cofactor = 131071; // large prime
        let mut r2 = make_rel(2, 1, vec![(0, 1)], vec![]);
        r2.rat_cofactor = 131071; // same large prime (appears twice)
        let mut r3 = make_rel(3, 1, vec![(0, 1)], vec![]);
        r3.rat_cofactor = 999983; // different large prime (singleton)

        let filtered = filter_relations(vec![r1, r2, r3]);
        assert_eq!(filtered.len(), 2);
    }

    #[test]
    fn test_alg_lp_key_distinguishes_roots() {
        // Same large-prime value p=11 but different ideal roots.
        let mut r1 = make_rel(1, 1, vec![(0, 1)], vec![]);
        let mut r2 = make_rel(2, 1, vec![(0, 1)], vec![]);
        let mut r3 = make_rel(13, 1, vec![(0, 1)], vec![]);
        r1.alg_cofactor = 11;
        r2.alg_cofactor = 11;
        r3.alg_cofactor = 11;

        // (a,b) = (1,1) -> r=1, (2,1)->r=2, so relation r1 is singleton and removed.
        let filtered = filter_relations(vec![r1, r2, r3]);
        assert_eq!(filtered.len(), 2);
        assert!(filtered.iter().all(|r| (r.a % 11 + 11) % 11 == 2));
    }

    #[test]
    fn test_alg_lp_projective_key_supported() {
        let mut r1 = make_rel(5, 14, vec![(0, 1)], vec![]);
        let mut r2 = make_rel(6, 21, vec![(0, 1)], vec![]);
        let mut r3 = make_rel(1, 1, vec![(0, 1)], vec![]);
        r1.alg_cofactor = 7;
        r2.alg_cofactor = 7;
        r3.alg_cofactor = 7;

        // r1,r2 have b % 7 == 0 => projective key (7,7), so they survive together.
        // r3 has affine key (7,1) and is singleton, so removed.
        let filtered = filter_relations(vec![r1, r2, r3]);
        assert_eq!(filtered.len(), 2);
        assert!(filtered.iter().all(|r| r.b % 7 == 0));
    }
}
