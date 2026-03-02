//! Filtering: remove duplicate and singleton relations.

use std::collections::{HashMap, HashSet};

use crate::relation::Relation;

/// Remove duplicate (a,b) pairs and relations containing singleton ideals.
/// A singleton is an ideal (prime column) that appears in only one relation.
/// Relations containing singletons are useless for LA and can be removed,
/// potentially creating new singletons. Iterate until stable.
pub fn filter_relations(relations: Vec<Relation>) -> Vec<Relation> {
    // Step 1: Deduplicate by (a, b)
    let mut seen = HashSet::new();
    let mut unique: Vec<Relation> = relations
        .into_iter()
        .filter(|r| seen.insert((r.a, r.b)))
        .collect();

    // Step 2: Iterative singleton removal
    loop {
        // Count how many times each column appears
        let mut col_weight: HashMap<u64, u32> = HashMap::new();
        for rel in &unique {
            for &(idx, _) in &rel.rational_factors {
                *col_weight.entry(idx as u64).or_default() += 1;
            }
            for &(idx, _) in &rel.algebraic_factors {
                *col_weight.entry(1_000_000 + idx as u64).or_default() += 1;
            }
            if rel.rat_cofactor > 1 {
                *col_weight.entry(2_000_000 + rel.rat_cofactor).or_default() += 1;
            }
            if rel.alg_cofactor > 1 {
                *col_weight.entry(3_000_000 + rel.alg_cofactor).or_default() += 1;
            }
        }

        let before = unique.len();
        unique.retain(|rel| {
            // Keep only if no column in this relation is a singleton
            let has_singleton = rel
                .rational_factors
                .iter()
                .any(|&(idx, _)| col_weight.get(&(idx as u64)).copied().unwrap_or(0) < 2)
                || rel.algebraic_factors.iter().any(|&(idx, _)| {
                    col_weight
                        .get(&(1_000_000 + idx as u64))
                        .copied()
                        .unwrap_or(0)
                        < 2
                })
                || (rel.rat_cofactor > 1
                    && col_weight
                        .get(&(2_000_000 + rel.rat_cofactor))
                        .copied()
                        .unwrap_or(0)
                        < 2)
                || (rel.alg_cofactor > 1
                    && col_weight
                        .get(&(3_000_000 + rel.alg_cofactor))
                        .copied()
                        .unwrap_or(0)
                        < 2);
            !has_singleton
        });

        if unique.len() == before {
            break;
        }
    }

    unique
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
            rat_cofactor: 0,
            alg_cofactor: 0,
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
    fn test_singleton_removal() {
        // Relations sharing column 0, but column 1 only appears once
        let rels = vec![
            make_rel(1, 1, vec![(0, 1), (1, 1)], vec![(0, 1)]), // col 1 is singleton
            make_rel(2, 1, vec![(0, 1)], vec![(0, 1)]),
            make_rel(3, 1, vec![(0, 1)], vec![(0, 1)]),
        ];
        let filtered = filter_relations(rels);
        // Rel 1 has singleton col 1, gets removed. Rels 2 and 3 share col 0.
        assert_eq!(filtered.len(), 2);
    }

    #[test]
    fn test_cascading_singleton() {
        // Removing one singleton creates another
        let rels = vec![
            make_rel(1, 1, vec![(0, 1), (1, 1)], vec![]), // col 0, col 1
            make_rel(2, 1, vec![(1, 1), (2, 1)], vec![]), // col 1, col 2
            make_rel(3, 1, vec![(2, 1), (3, 1)], vec![]), // col 2, col 3
            make_rel(4, 1, vec![(3, 1)], vec![]),          // col 3
        ];
        // col 0 is singleton -> remove rel 1
        // now col 1 is singleton -> remove rel 2
        // now col 2 is singleton -> remove rel 3
        // now col 3 is singleton -> remove rel 4
        let filtered = filter_relations(rels);
        assert_eq!(filtered.len(), 0);
    }

    #[test]
    fn test_all_shared() {
        // All columns appear at least twice
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
    fn test_large_prime_columns() {
        // Relations with large prime cofactors
        let mut r1 = make_rel(1, 1, vec![(0, 1)], vec![]);
        r1.rat_cofactor = 131071; // large prime
        let mut r2 = make_rel(2, 1, vec![(0, 1)], vec![]);
        r2.rat_cofactor = 131071; // same large prime (appears twice)
        let mut r3 = make_rel(3, 1, vec![(0, 1)], vec![]);
        r3.rat_cofactor = 999983; // different large prime (singleton)

        let filtered = filter_relations(vec![r1, r2, r3]);
        // r3 has a singleton large prime column -> removed
        assert_eq!(filtered.len(), 2);
    }
}
