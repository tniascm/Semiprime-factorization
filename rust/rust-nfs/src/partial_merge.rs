//! Deterministic 2LP graph merge:
//! build relation sets whose LP-key parity is zero.

use std::collections::{HashMap, HashSet, VecDeque};

use crate::lp_key::LpKey;
use crate::relation::Relation;

#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct PartialMergeStats {
    pub total_relations: usize,
    pub relations_0lp: usize,
    pub relations_1lp: usize,
    pub relations_2lp: usize,
    pub relations_dropped_gt2lp: usize,
    pub lp_nodes: usize,
    pub tree_edges: usize,
    pub cycles_found: usize,
    pub output_sets: usize,
}

#[derive(Debug, Default)]
struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<u8>,
}

impl UnionFind {
    fn with_nodes(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0u8; n],
        }
    }

    fn add_node(&mut self) -> usize {
        let id = self.parent.len();
        self.parent.push(id);
        self.rank.push(0);
        id
    }

    fn find(&mut self, mut x: usize) -> usize {
        let mut root = x;
        while self.parent[root] != root {
            root = self.parent[root];
        }
        while self.parent[x] != x {
            let p = self.parent[x];
            self.parent[x] = root;
            x = p;
        }
        root
    }

    fn union(&mut self, a: usize, b: usize) {
        let mut ra = self.find(a);
        let mut rb = self.find(b);
        if ra == rb {
            return;
        }
        if self.rank[ra] < self.rank[rb] {
            std::mem::swap(&mut ra, &mut rb);
        }
        self.parent[rb] = ra;
        if self.rank[ra] == self.rank[rb] {
            self.rank[ra] = self.rank[ra].saturating_add(1);
        }
    }
}

fn normalize_lp_keys(keys: &[LpKey]) -> Vec<LpKey> {
    // Keep odd multiplicity only (GF(2) parity), deterministic order.
    let mut parity: HashSet<LpKey> = HashSet::new();
    for &k in keys {
        if !parity.remove(&k) {
            parity.insert(k);
        }
    }
    let mut out: Vec<LpKey> = parity.into_iter().collect();
    out.sort_unstable();
    out
}

fn get_or_insert_node(
    key: LpKey,
    map: &mut HashMap<LpKey, usize>,
    uf: &mut UnionFind,
    tree_adj: &mut Vec<Vec<(usize, usize)>>,
) -> usize {
    if let Some(&id) = map.get(&key) {
        return id;
    }
    let id = uf.add_node();
    tree_adj.push(Vec::new());
    map.insert(key, id);
    id
}

fn tree_path_relation_indices(
    tree_adj: &[Vec<(usize, usize)>],
    src: usize,
    dst: usize,
) -> Option<Vec<usize>> {
    if src == dst {
        return Some(Vec::new());
    }
    if src >= tree_adj.len() || dst >= tree_adj.len() {
        return None;
    }
    let n = tree_adj.len();
    let mut prev_node = vec![usize::MAX; n];
    let mut prev_edge = vec![usize::MAX; n];
    let mut q = VecDeque::new();

    prev_node[src] = src;
    q.push_back(src);

    while let Some(u) = q.pop_front() {
        if u == dst {
            break;
        }
        for &(v, rel_idx) in &tree_adj[u] {
            if prev_node[v] != usize::MAX {
                continue;
            }
            prev_node[v] = u;
            prev_edge[v] = rel_idx;
            q.push_back(v);
        }
    }

    if prev_node[dst] == usize::MAX {
        return None;
    }

    let mut edges = Vec::new();
    let mut cur = dst;
    while cur != src {
        edges.push(prev_edge[cur]);
        cur = prev_node[cur];
    }
    edges.reverse();
    Some(edges)
}

pub fn merge_relations_2lp(
    relations: &[Relation],
    max_sets: usize,
) -> (Vec<Vec<usize>>, PartialMergeStats) {
    let mut stats = PartialMergeStats {
        total_relations: relations.len(),
        ..PartialMergeStats::default()
    };
    if max_sets == 0 || relations.is_empty() {
        return (Vec::new(), stats);
    }

    // Node 0 is the synthetic sink for 1LP edges.
    let sink = 0usize;
    let mut uf = UnionFind::with_nodes(1);
    let mut key_to_node: HashMap<LpKey, usize> = HashMap::new();
    let mut tree_adj: Vec<Vec<(usize, usize)>> = vec![Vec::new()];

    let mut sets: Vec<Vec<usize>> = Vec::new();
    let mut seen_sets: HashSet<Vec<usize>> = HashSet::new();

    for (rel_idx, rel) in relations.iter().enumerate() {
        if sets.len() >= max_sets {
            break;
        }

        let keys = normalize_lp_keys(&rel.lp_keys);
        match keys.len() {
            0 => {
                stats.relations_0lp += 1;
                let singleton = vec![rel_idx];
                if seen_sets.insert(singleton.clone()) {
                    sets.push(singleton);
                }
            }
            1 => {
                stats.relations_1lp += 1;
                let u = get_or_insert_node(keys[0], &mut key_to_node, &mut uf, &mut tree_adj);
                let v = sink;

                if uf.find(u) != uf.find(v) {
                    uf.union(u, v);
                    tree_adj[u].push((v, rel_idx));
                    tree_adj[v].push((u, rel_idx));
                    stats.tree_edges += 1;
                } else {
                    stats.cycles_found += 1;
                    if let Some(mut path) = tree_path_relation_indices(&tree_adj, u, v) {
                        path.push(rel_idx);
                        path.sort_unstable();
                        path.dedup();
                        if seen_sets.insert(path.clone()) {
                            sets.push(path);
                        }
                    }
                }
            }
            2 => {
                stats.relations_2lp += 1;
                let u = get_or_insert_node(keys[0], &mut key_to_node, &mut uf, &mut tree_adj);
                let v = get_or_insert_node(keys[1], &mut key_to_node, &mut uf, &mut tree_adj);

                if uf.find(u) != uf.find(v) {
                    uf.union(u, v);
                    tree_adj[u].push((v, rel_idx));
                    tree_adj[v].push((u, rel_idx));
                    stats.tree_edges += 1;
                } else {
                    stats.cycles_found += 1;
                    if let Some(mut path) = tree_path_relation_indices(&tree_adj, u, v) {
                        path.push(rel_idx);
                        path.sort_unstable();
                        path.dedup();
                        if seen_sets.insert(path.clone()) {
                            sets.push(path);
                        }
                    }
                }
            }
            _ => {
                stats.relations_dropped_gt2lp += 1;
            }
        }
    }

    stats.lp_nodes = key_to_node.len();
    stats.output_sets = sets.len();
    (sets, stats)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rel_with_keys(keys: Vec<LpKey>) -> Relation {
        Relation {
            a: 1,
            b: 1,
            rational_factors: vec![],
            algebraic_factors: vec![],
            rational_sign_negative: false,
            algebraic_sign_negative: false,
            special_q: None,
            rat_cofactor: 0,
            alg_cofactor: 0,
            lp_keys: keys,
        }
    }

    #[test]
    fn test_merge_relations_2lp_finds_basic_cycles() {
        let a = LpKey::Rational(101);
        let b = LpKey::Rational(103);
        let rels = vec![
            rel_with_keys(vec![]),     // 0LP -> immediate set [0]
            rel_with_keys(vec![a]),    // tree edge a-sink
            rel_with_keys(vec![a]),    // closes cycle with previous -> [1,2]
            rel_with_keys(vec![a, b]), // tree edge a-b
            rel_with_keys(vec![b]),    // closes cycle b-sink via (a-b)+(a-sink) -> [1,3,4]
        ];

        let (sets, stats) = merge_relations_2lp(&rels, 100);
        let as_set: HashSet<Vec<usize>> = sets.into_iter().collect();

        assert!(as_set.contains(&vec![0]));
        assert!(as_set.contains(&vec![1, 2]));
        assert!(as_set.contains(&vec![1, 3, 4]));
        assert_eq!(stats.relations_0lp, 1);
        assert_eq!(stats.relations_1lp, 3);
        assert_eq!(stats.relations_2lp, 1);
        assert_eq!(stats.relations_dropped_gt2lp, 0);
        assert!(stats.cycles_found >= 2);
    }
}
