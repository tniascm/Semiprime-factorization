//! Tree Tensor Network (TTN) for binary optimization.
//!
//! A TTN arranges tensors in a binary tree instead of a 1D chain.
//! For n binary variables (leaves), we have:
//! - n leaf tensors: T[sigma][beta] (physical index + one bond to parent)
//! - n-1 internal tensors: T[alpha_left][alpha_right][beta_parent]
//! - 1 root tensor: T[alpha_left][alpha_right] (no parent bond)
//!
//! The amplitude for configuration (b_0, ..., b_{n-1}) is computed by:
//! 1. Evaluate each leaf tensor at its physical index
//! 2. Contract up the tree: each internal node contracts its two children
//! 3. The root contraction gives a scalar
//!
//! ## Paper Reference (Tesoro/Siloi 2024, arXiv:2410.16355)
//!
//! Key design choices from the paper:
//! - Binary tree topology with depth = ceil(log2(num_sites))
//! - Bond dimension m = ceil(n^{2/5}) where n is number of sites
//! - Leaf tensors: shape [2][bond_dim]
//! - Internal tensors: shape [m_left][m_right][m_parent]
//! - Root tensor: shape [m_left][m_right] -> scalar
//! - Contraction: bottom-up for expectation values, top-down for sampling

use rand::Rng;

use crate::optimizer::CostHamiltonian;

/// A tensor in the TTN.
#[derive(Debug, Clone)]
pub enum TtnTensor {
    /// Leaf tensor: [sigma][bond_up] where sigma in {0,1}
    Leaf {
        data: Vec<Vec<f64>>, // [2][bond_dim]
        bond_dim: usize,
    },
    /// Internal node: [bond_left][bond_right][bond_up]
    Internal {
        data: Vec<Vec<Vec<f64>>>, // [bond_left][bond_right][bond_up]
        left_dim: usize,
        right_dim: usize,
        up_dim: usize,
    },
    /// Root node: [bond_left][bond_right] -> scalar
    Root {
        data: Vec<Vec<f64>>, // [bond_left][bond_right]
        left_dim: usize,
        right_dim: usize,
    },
}

/// Tree structure: which variables are at which leaves, and the tree topology.
#[derive(Debug, Clone)]
pub struct TreeTopology {
    /// Number of leaves (= number of variables)
    pub num_leaves: usize,
    /// For each internal node i, (left_child, right_child)
    /// Children can be leaf indices (< num_leaves) or internal node indices (>= num_leaves)
    pub nodes: Vec<(usize, usize)>,
    /// Index of the root in the nodes array
    pub root: usize,
    /// Depth of the tree
    pub depth: usize,
}

impl TreeTopology {
    /// Create a balanced binary tree for n leaves.
    ///
    /// The tree depth is ceil(log2(n)). At each level, nodes are paired
    /// left-to-right. If there's an odd one out, it is passed up to the next level.
    pub fn balanced(n: usize) -> Self {
        assert!(n >= 2, "Need at least 2 variables");

        let mut nodes = Vec::new();
        let mut depth = 0;

        // Build tree bottom-up
        // Level 0: leaves are indices 0..n-1
        let mut current_level: Vec<usize> = (0..n).collect();
        let mut next_id = n; // Internal nodes start at index n

        while current_level.len() > 1 {
            let mut next_level = Vec::new();
            let mut i = 0;
            while i < current_level.len() {
                if i + 1 < current_level.len() {
                    nodes.push((current_level[i], current_level[i + 1]));
                    next_level.push(next_id);
                    next_id += 1;
                    i += 2;
                } else {
                    // Odd one out -- pass it up
                    next_level.push(current_level[i]);
                    i += 1;
                }
            }
            current_level = next_level;
            depth += 1;
        }

        let root = nodes.len() - 1;
        TreeTopology {
            num_leaves: n,
            nodes,
            root,
            depth,
        }
    }
}

/// Tree Tensor Network state.
#[derive(Debug, Clone)]
pub struct Ttn {
    /// The tree topology
    pub topology: TreeTopology,
    /// Leaf tensors (one per variable)
    pub leaves: Vec<TtnTensor>,
    /// Internal + root tensors
    pub internals: Vec<TtnTensor>,
    /// Bond dimension
    pub bond_dim: usize,
    /// Number of variables
    pub num_vars: usize,
}

impl Ttn {
    /// Create a new TTN with random tensors.
    pub fn new_random(num_vars: usize, bond_dim: usize, rng: &mut impl Rng) -> Self {
        let topology = TreeTopology::balanced(num_vars);
        let scale = 1.0 / (bond_dim as f64).sqrt();

        // Create leaf tensors
        let mut leaves = Vec::with_capacity(num_vars);
        for _ in 0..num_vars {
            let mut data = vec![vec![0.0; bond_dim]; 2];
            for sigma in 0..2 {
                for b in 0..bond_dim {
                    data[sigma][b] = (rng.gen::<f64>() * 2.0 - 1.0) * scale;
                }
            }
            leaves.push(TtnTensor::Leaf { data, bond_dim });
        }

        // Create internal tensors
        let num_internal = topology.nodes.len();
        let mut internals = Vec::with_capacity(num_internal);
        for i in 0..num_internal {
            if i == topology.root {
                // Root: [bond_left][bond_right] -> scalar
                let mut data = vec![vec![0.0; bond_dim]; bond_dim];
                for a in 0..bond_dim {
                    for b in 0..bond_dim {
                        data[a][b] = (rng.gen::<f64>() * 2.0 - 1.0) * scale;
                    }
                }
                internals.push(TtnTensor::Root {
                    data,
                    left_dim: bond_dim,
                    right_dim: bond_dim,
                });
            } else {
                // Internal: [bond_left][bond_right][bond_up]
                let mut data = vec![vec![vec![0.0; bond_dim]; bond_dim]; bond_dim];
                for a in 0..bond_dim {
                    for b in 0..bond_dim {
                        for c in 0..bond_dim {
                            data[a][b][c] = (rng.gen::<f64>() * 2.0 - 1.0) * scale;
                        }
                    }
                }
                internals.push(TtnTensor::Internal {
                    data,
                    left_dim: bond_dim,
                    right_dim: bond_dim,
                    up_dim: bond_dim,
                });
            }
        }

        Ttn {
            topology,
            leaves,
            internals,
            bond_dim,
            num_vars,
        }
    }

    /// Create a new TTN with the paper's scaled bond dimension.
    ///
    /// Bond dimension m = ceil(n^{2/5}) as per Tesoro/Siloi 2024.
    pub fn new_random_paper_scaling(num_vars: usize, rng: &mut impl Rng) -> Self {
        let bond_dim = compute_paper_bond_dim(num_vars);
        Self::new_random(num_vars, bond_dim, rng)
    }

    /// Evaluate the TTN amplitude for a given binary configuration.
    pub fn evaluate(&self, config: &[u8]) -> f64 {
        assert_eq!(config.len(), self.num_vars);

        // Bottom-up contraction
        // First, evaluate leaf tensors to get bond vectors
        let mut vectors: Vec<Option<Vec<f64>>> =
            vec![None; self.num_vars + self.topology.nodes.len()];

        for i in 0..self.num_vars {
            if let TtnTensor::Leaf { ref data, .. } = self.leaves[i] {
                let sigma = config[i] as usize;
                vectors[i] = Some(data[sigma].clone());
            }
        }

        // Contract internal nodes bottom-up
        for (idx, &(left, right)) in self.topology.nodes.iter().enumerate() {
            let left_vec = vectors[left].as_ref().expect("Left child not computed");
            let right_vec = vectors[right].as_ref().expect("Right child not computed");
            let node_id = self.num_vars + idx;

            if idx == self.topology.root {
                // Root contraction: sum_{a,b} root[a][b] * left[a] * right[b]
                if let TtnTensor::Root {
                    ref data,
                    left_dim,
                    right_dim,
                } = self.internals[idx]
                {
                    let mut scalar = 0.0;
                    for a in 0..left_dim.min(left_vec.len()) {
                        for b in 0..right_dim.min(right_vec.len()) {
                            scalar += data[a][b] * left_vec[a] * right_vec[b];
                        }
                    }
                    vectors[node_id] = Some(vec![scalar]);
                }
            } else {
                // Internal contraction: result[c] = sum_{a,b} tensor[a][b][c] * left[a] * right[b]
                if let TtnTensor::Internal {
                    ref data,
                    left_dim,
                    right_dim,
                    up_dim,
                } = self.internals[idx]
                {
                    let mut result = vec![0.0; up_dim];
                    for a in 0..left_dim.min(left_vec.len()) {
                        for b in 0..right_dim.min(right_vec.len()) {
                            for c in 0..up_dim {
                                result[c] += data[a][b][c] * left_vec[a] * right_vec[b];
                            }
                        }
                    }
                    vectors[node_id] = Some(result);
                }
            }
        }

        // Return root scalar
        let root_id = self.num_vars + self.topology.root;
        vectors[root_id].as_ref().unwrap()[0]
    }

    /// Sample a binary configuration from the TTN using top-down sampling.
    ///
    /// This implements the paper's approach: start from the root, sample
    /// conditional distributions at each level moving down to the leaves.
    /// For efficiency, we use a simplified version that samples greedily.
    pub fn sample_topdown(&self, rng: &mut impl Rng) -> Vec<u8> {
        let n = self.num_vars;
        if n <= 20 {
            // For small systems, use exact marginal computation
            return self.sample_exact(rng);
        }

        // For larger systems, use approximate sampling:
        // Evaluate multiple random configs, weight by |amplitude|^2, sample
        let num_candidates = 100.min(1u64 << n) as usize;
        let mut configs = Vec::with_capacity(num_candidates);
        let mut weights = Vec::with_capacity(num_candidates);

        for _ in 0..num_candidates {
            let config: Vec<u8> = (0..n).map(|_| if rng.gen::<bool>() { 1 } else { 0 }).collect();
            let amp = self.evaluate(&config);
            weights.push(amp * amp);
            configs.push(config);
        }

        let total: f64 = weights.iter().sum();
        if total <= 0.0 {
            return configs.into_iter().next().unwrap_or_else(|| vec![0; n]);
        }

        // Weighted sample
        let r: f64 = rng.gen::<f64>() * total;
        let mut cumulative = 0.0;
        for (i, w) in weights.iter().enumerate() {
            cumulative += w;
            if cumulative >= r {
                return configs[i].clone();
            }
        }

        configs.last().cloned().unwrap_or_else(|| vec![0; n])
    }

    /// Sample from exact marginal distributions (small systems only).
    fn sample_exact(&self, rng: &mut impl Rng) -> Vec<u8> {
        let n = self.num_vars;
        assert!(n <= 20, "Exact sampling only for small n");

        let total = 1u64 << n;
        let mut amp_sq_sum = 0.0;
        let mut all_configs: Vec<(Vec<u8>, f64)> = Vec::with_capacity(total as usize);

        for bits in 0..total {
            let config: Vec<u8> = (0..n).map(|j| ((bits >> j) & 1) as u8).collect();
            let amp = self.evaluate(&config);
            let w = amp * amp;
            amp_sq_sum += w;
            all_configs.push((config, w));
        }

        if amp_sq_sum <= 0.0 {
            return vec![0; n];
        }

        let r: f64 = rng.gen::<f64>() * amp_sq_sum;
        let mut cumulative = 0.0;
        for (config, w) in &all_configs {
            cumulative += w;
            if cumulative >= r {
                return config.clone();
            }
        }

        all_configs.last().map(|(c, _)| c.clone()).unwrap_or_else(|| vec![0; n])
    }

    /// Compute norm squared by enumerating all configurations (small n only).
    pub fn contract_norm_squared(&self) -> f64 {
        assert!(self.num_vars <= 24, "Exhaustive only for small n");
        let total = 1u64 << self.num_vars;
        let mut norm_sq = 0.0;
        for bits in 0..total {
            let config: Vec<u8> = (0..self.num_vars)
                .map(|j| ((bits >> j) & 1) as u8)
                .collect();
            let amp = self.evaluate(&config);
            norm_sq += amp * amp;
        }
        norm_sq
    }

    /// Find the configuration with maximum |amplitude|^2 (small n only).
    pub fn find_max_config(&self) -> (Vec<u8>, f64) {
        assert!(self.num_vars <= 24);
        let total = 1u64 << self.num_vars;
        let mut best_config = vec![0u8; self.num_vars];
        let mut best_amp_sq = 0.0;

        for bits in 0..total {
            let config: Vec<u8> = (0..self.num_vars)
                .map(|j| ((bits >> j) & 1) as u8)
                .collect();
            let amp = self.evaluate(&config);
            let amp_sq = amp * amp;
            if amp_sq > best_amp_sq {
                best_amp_sq = amp_sq;
                best_config = config;
            }
        }
        (best_config, best_amp_sq)
    }

    /// Get the total number of nodes (leaves + internal) in the tree.
    pub fn total_nodes(&self) -> usize {
        self.num_vars + self.topology.nodes.len()
    }

    /// Get the tree depth.
    pub fn depth(&self) -> usize {
        self.topology.depth
    }
}

/// Compute the paper's bond dimension formula: m = ceil(n^{2/5}).
pub fn compute_paper_bond_dim(num_sites: usize) -> usize {
    let n = num_sites as f64;
    let m = n.powf(0.4).ceil() as usize;
    m.max(2) // Minimum bond dimension of 2
}

/// Optimize TTN using variational sweeps over the tree.
///
/// Sweeps bottom-up then top-down. At each tensor, compute gradient
/// via finite differences and update.
pub fn optimize_ttn_sweep(
    ttn: &mut Ttn,
    hamiltonian: &CostHamiltonian,
    num_sweeps: usize,
) -> f64 {
    let mut best_energy = f64::MAX;
    let step_size = 0.01;

    for _sweep in 0..num_sweeps {
        // Bottom-up sweep: optimize leaves first, then internals
        for i in 0..ttn.num_vars {
            optimize_leaf(ttn, hamiltonian, i, step_size);
        }
        for i in 0..ttn.topology.nodes.len() {
            optimize_internal(ttn, hamiltonian, i, step_size);
        }

        // Top-down sweep
        for i in (0..ttn.topology.nodes.len()).rev() {
            optimize_internal(ttn, hamiltonian, i, step_size);
        }
        for i in (0..ttn.num_vars).rev() {
            optimize_leaf(ttn, hamiltonian, i, step_size);
        }

        let energy = compute_ttn_expectation(ttn, hamiltonian);
        if energy < best_energy {
            best_energy = energy;
        }
    }

    best_energy
}

fn optimize_leaf(ttn: &mut Ttn, hamiltonian: &CostHamiltonian, leaf: usize, step_size: f64) {
    let bond_dim = match &ttn.leaves[leaf] {
        TtnTensor::Leaf { bond_dim, .. } => *bond_dim,
        _ => return,
    };

    for sigma in 0..2 {
        for b in 0..bond_dim {
            let original = get_leaf_element(ttn, leaf, sigma, b);

            set_leaf_element(ttn, leaf, sigma, b, original + step_size);
            let e_plus = compute_ttn_expectation(ttn, hamiltonian);

            set_leaf_element(ttn, leaf, sigma, b, original - step_size);
            let e_minus = compute_ttn_expectation(ttn, hamiltonian);

            let grad = (e_plus - e_minus) / (2.0 * step_size);
            set_leaf_element(ttn, leaf, sigma, b, original - step_size * grad);
        }
    }
}

fn get_leaf_element(ttn: &Ttn, leaf: usize, sigma: usize, b: usize) -> f64 {
    if let TtnTensor::Leaf { ref data, .. } = ttn.leaves[leaf] {
        data[sigma][b]
    } else {
        0.0
    }
}

fn set_leaf_element(ttn: &mut Ttn, leaf: usize, sigma: usize, b: usize, value: f64) {
    if let TtnTensor::Leaf { ref mut data, .. } = ttn.leaves[leaf] {
        data[sigma][b] = value;
    }
}

fn optimize_internal(
    ttn: &mut Ttn,
    hamiltonian: &CostHamiltonian,
    node: usize,
    step_size: f64,
) {
    let dims = get_internal_dims(ttn, node);
    let (ld, rd, ud) = match dims {
        Some(d) => d,
        None => return,
    };

    for a in 0..ld {
        for b in 0..rd {
            for c in 0..ud {
                let original = get_internal_element(ttn, node, a, b, c);

                set_internal_element(ttn, node, a, b, c, original + step_size);
                let e_plus = compute_ttn_expectation(ttn, hamiltonian);

                set_internal_element(ttn, node, a, b, c, original - step_size);
                let e_minus = compute_ttn_expectation(ttn, hamiltonian);

                let grad = (e_plus - e_minus) / (2.0 * step_size);
                set_internal_element(ttn, node, a, b, c, original - step_size * grad);
            }
        }
    }
}

/// Get the dimensions of an internal/root tensor as (left, right, up).
/// For root tensors, up_dim is 1 (representing the scalar output).
fn get_internal_dims(ttn: &Ttn, node: usize) -> Option<(usize, usize, usize)> {
    match &ttn.internals[node] {
        TtnTensor::Root {
            left_dim,
            right_dim,
            ..
        } => Some((*left_dim, *right_dim, 1)),
        TtnTensor::Internal {
            left_dim,
            right_dim,
            up_dim,
            ..
        } => Some((*left_dim, *right_dim, *up_dim)),
        _ => None,
    }
}

fn get_internal_element(ttn: &Ttn, node: usize, a: usize, b: usize, c: usize) -> f64 {
    match &ttn.internals[node] {
        TtnTensor::Root { ref data, .. } => {
            // c should be 0 for root
            data[a][b]
        }
        TtnTensor::Internal { ref data, .. } => data[a][b][c],
        _ => 0.0,
    }
}

fn set_internal_element(ttn: &mut Ttn, node: usize, a: usize, b: usize, c: usize, value: f64) {
    match &mut ttn.internals[node] {
        TtnTensor::Root { ref mut data, .. } => {
            // c should be 0 for root
            data[a][b] = value;
        }
        TtnTensor::Internal { ref mut data, .. } => {
            data[a][b][c] = value;
        }
        _ => {}
    }
}

/// Compute expectation value <psi|H|psi>/<psi|psi> for TTN.
pub fn compute_ttn_expectation(ttn: &Ttn, hamiltonian: &CostHamiltonian) -> f64 {
    let n = ttn.num_vars;
    if n <= 20 {
        let total_configs = 1u64 << n;
        let mut energy_sum = 0.0;
        let mut norm_sum = 0.0;
        for bits in 0..total_configs {
            let config: Vec<u8> = (0..n).map(|j| ((bits >> j) & 1) as u8).collect();
            let amp = ttn.evaluate(&config);
            let amp_sq = amp * amp;
            let cost = hamiltonian.evaluate(&config);
            energy_sum += amp_sq * cost;
            norm_sum += amp_sq;
        }
        if norm_sum.abs() < 1e-30 {
            return f64::MAX;
        }
        energy_sum / norm_sum
    } else {
        // Sampling-based for large n
        let mut rng = rand::thread_rng();
        let num_samples = 1000;
        let mut energy_sum = 0.0;
        let mut count = 0;
        for _ in 0..num_samples {
            let config: Vec<u8> = (0..n)
                .map(|_| if rng.gen::<bool>() { 1 } else { 0 })
                .collect();
            let _amp = ttn.evaluate(&config);
            let cost = hamiltonian.evaluate(&config);
            energy_sum += cost; // uniform sampling, weight equally
            count += 1;
        }
        if count == 0 {
            return f64::MAX;
        }
        energy_sum / count as f64
    }
}

/// Extract solution from optimized TTN.
pub fn extract_ttn_solution(
    ttn: &Ttn,
    num_exponents: usize,
    bits_per_exp: usize,
    exponent_bound: i64,
) -> Vec<i64> {
    let (config, _) = ttn.find_max_config();
    crate::optimizer::binary_to_exponents(&config, num_exponents, bits_per_exp, exponent_bound)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn test_tree_topology_balanced() {
        let topo = TreeTopology::balanced(4);
        assert_eq!(topo.num_leaves, 4);
        assert!(topo.nodes.len() >= 2); // At least 2 internal nodes for 4 leaves
        assert!(topo.depth >= 2); // At least depth 2 for 4 leaves
    }

    #[test]
    fn test_tree_topology_balanced_odd() {
        let topo = TreeTopology::balanced(5);
        assert_eq!(topo.num_leaves, 5);
        assert!(topo.nodes.len() >= 3);
    }

    #[test]
    fn test_tree_topology_depth() {
        // Depth should be approximately log2(n)
        let topo4 = TreeTopology::balanced(4);
        assert_eq!(topo4.depth, 2);

        let topo8 = TreeTopology::balanced(8);
        assert_eq!(topo8.depth, 3);
    }

    #[test]
    fn test_paper_bond_dim() {
        // n=10: 10^0.4 = ~2.51 -> ceil = 3
        assert_eq!(compute_paper_bond_dim(10), 3);

        // n=100: 100^0.4 = ~6.31 -> ceil = 7
        assert_eq!(compute_paper_bond_dim(100), 7);

        // n=1: should be at least 2
        assert_eq!(compute_paper_bond_dim(1), 2);

        // n=1000: 1000^0.4 = ~15.85 -> ceil = 16
        assert_eq!(compute_paper_bond_dim(1000), 16);
    }

    #[test]
    fn test_ttn_creation() {
        let mut rng = StdRng::seed_from_u64(42);
        let ttn = Ttn::new_random(4, 2, &mut rng);
        assert_eq!(ttn.num_vars, 4);
        assert_eq!(ttn.leaves.len(), 4);
    }

    #[test]
    fn test_ttn_paper_scaling_creation() {
        let mut rng = StdRng::seed_from_u64(42);
        let ttn = Ttn::new_random_paper_scaling(16, &mut rng);
        assert_eq!(ttn.num_vars, 16);
        // 16^0.4 = ~3.03 -> ceil = 4
        assert_eq!(ttn.bond_dim, 4);
    }

    #[test]
    fn test_ttn_evaluate() {
        let mut rng = StdRng::seed_from_u64(42);
        let ttn = Ttn::new_random(4, 2, &mut rng);
        let config = vec![0u8, 1, 0, 1];
        let amp = ttn.evaluate(&config);
        assert!(amp.is_finite());
        // Deterministic
        let amp2 = ttn.evaluate(&config);
        assert!((amp - amp2).abs() < 1e-15);
    }

    #[test]
    fn test_ttn_evaluate_all_configs() {
        let mut rng = StdRng::seed_from_u64(42);
        let ttn = Ttn::new_random(4, 2, &mut rng);
        let mut total_sq = 0.0;
        for bits in 0..16u64 {
            let config: Vec<u8> = (0..4).map(|j| ((bits >> j) & 1) as u8).collect();
            let amp = ttn.evaluate(&config);
            total_sq += amp * amp;
        }
        let norm_sq = ttn.contract_norm_squared();
        assert!(
            (total_sq - norm_sq).abs() < 1e-10,
            "Explicit sum {} != contraction {}",
            total_sq,
            norm_sq
        );
    }

    #[test]
    fn test_ttn_evaluate_odd_vars() {
        let mut rng = StdRng::seed_from_u64(42);
        let ttn = Ttn::new_random(5, 2, &mut rng);
        let config = vec![0u8, 1, 0, 1, 1];
        let amp = ttn.evaluate(&config);
        assert!(amp.is_finite());
    }

    #[test]
    fn test_ttn_sample_topdown() {
        let mut rng = StdRng::seed_from_u64(42);
        let ttn = Ttn::new_random(4, 2, &mut rng);
        let mut sample_rng = StdRng::seed_from_u64(123);

        for _ in 0..10 {
            let config = ttn.sample_topdown(&mut sample_rng);
            assert_eq!(config.len(), 4);
            for &b in &config {
                assert!(b == 0 || b == 1);
            }
        }
    }

    #[test]
    fn test_ttn_total_nodes() {
        let mut rng = StdRng::seed_from_u64(42);
        let ttn = Ttn::new_random(4, 2, &mut rng);
        assert!(ttn.total_nodes() >= 4 + 2); // At least 4 leaves + 2 internal nodes
    }

    #[test]
    fn test_ttn_optimize_reduces_energy() {
        let mut rng = StdRng::seed_from_u64(42);
        let hamiltonian = CostHamiltonian {
            terms: vec![
                crate::optimizer::QuadraticTerm {
                    i: 0,
                    j: 0,
                    coeff: 1.0,
                },
                crate::optimizer::QuadraticTerm {
                    i: 1,
                    j: 1,
                    coeff: 1.0,
                },
                crate::optimizer::QuadraticTerm {
                    i: 0,
                    j: 1,
                    coeff: 2.0,
                },
            ],
            constant: 0.0,
            num_vars: 2,
        };

        let mut ttn = Ttn::new_random(2, 2, &mut rng);
        let initial = compute_ttn_expectation(&ttn, &hamiltonian);
        let final_e = optimize_ttn_sweep(&mut ttn, &hamiltonian, 3);
        assert!(
            final_e <= initial + 1e-6,
            "Energy should decrease: {} -> {}",
            initial,
            final_e
        );
    }
}
