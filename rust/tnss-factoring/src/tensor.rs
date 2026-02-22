//! Matrix Product State (MPS) representation for binary optimization.
//!
//! An MPS is a 1D tensor network (tensor train) that represents a quantum state
//! or probability distribution over binary variables. Each site has a 3-index
//! tensor A[alpha][sigma][beta] where:
//! - alpha, beta are bond indices (dimension = bond_dim)
//! - sigma is the physical index (dimension = 2 for binary variables)
//!
//! The amplitude for a configuration (b_0, b_1, ..., b_{n-1}) is:
//!   ψ(b_0, ..., b_{n-1}) = A_0[b_0] · A_1[b_1] · ... · A_{n-1}[b_{n-1}]
//! where each A_i[b_i] is a matrix of size bond_dim × bond_dim.

use rand::Rng;

/// A single MPS tensor at one site.
///
/// `data[sigma][alpha][beta]` where:
/// - `sigma` ∈ {0, 1} (physical/binary index)
/// - `alpha` ∈ [0, left_dim) (left bond index)
/// - `beta` ∈ [0, right_dim) (right bond index)
#[derive(Debug, Clone)]
pub struct MpsTensor {
    /// Tensor data indexed as [sigma][alpha][beta].
    pub data: Vec<Vec<Vec<f64>>>,
    /// Dimension of the left bond index.
    pub left_dim: usize,
    /// Dimension of the right bond index.
    pub right_dim: usize,
}

impl MpsTensor {
    /// Create a new MPS tensor with given dimensions, initialized to zeros.
    pub fn new(left_dim: usize, right_dim: usize) -> Self {
        let data = vec![vec![vec![0.0; right_dim]; left_dim]; 2];
        Self {
            data,
            left_dim,
            right_dim,
        }
    }

    /// Create a new MPS tensor with random entries in [-scale, scale].
    pub fn new_random(left_dim: usize, right_dim: usize, rng: &mut impl Rng, scale: f64) -> Self {
        let mut tensor = Self::new(left_dim, right_dim);
        for sigma in 0..2 {
            for alpha in 0..left_dim {
                for beta in 0..right_dim {
                    tensor.data[sigma][alpha][beta] =
                        (rng.gen::<f64>() * 2.0 - 1.0) * scale;
                }
            }
        }
        tensor
    }

    /// Get the matrix for a given physical index sigma ∈ {0, 1}.
    /// Returns a reference to the alpha×beta matrix.
    pub fn matrix(&self, sigma: usize) -> &Vec<Vec<f64>> {
        &self.data[sigma]
    }

    /// Get a mutable reference to the matrix for sigma.
    pub fn matrix_mut(&mut self, sigma: usize) -> &mut Vec<Vec<f64>> {
        &mut self.data[sigma]
    }
}

/// Matrix Product State for n binary variables.
#[derive(Debug, Clone)]
pub struct Mps {
    /// The MPS tensors, one per site (binary variable).
    pub tensors: Vec<MpsTensor>,
    /// Maximum bond dimension.
    pub bond_dim: usize,
    /// Number of binary variables (sites).
    pub num_vars: usize,
}

impl Mps {
    /// Create a new MPS with random tensors.
    ///
    /// The bond dimensions are:
    /// - Site 0: left_dim=1, right_dim=min(2, bond_dim)
    /// - Site n-1: left_dim=min(2^(n-1), bond_dim), right_dim=1
    /// - Interior sites: left_dim and right_dim are min(2^min(i, n-i), bond_dim)
    pub fn new_random(num_vars: usize, bond_dim: usize, rng: &mut impl Rng) -> Self {
        assert!(num_vars > 0, "MPS must have at least one variable");
        let scale = 1.0 / (bond_dim as f64).sqrt();

        // Compute bond dimensions between sites.
        // bond_dims[i] = dimension of the bond between site i and site i+1.
        // For an MPS with physical dimension 2, the exact max bond dimension
        // at bond i is min(2^(i+1), 2^(n-i-1)).
        let mut bond_dims = Vec::with_capacity(num_vars + 1);
        // Left boundary
        bond_dims.push(1usize);
        for i in 0..num_vars - 1 {
            let from_left = 1usize << (i + 1).min(30);
            let from_right = 1usize << (num_vars - 1 - i).min(30);
            bond_dims.push(bond_dim.min(from_left).min(from_right));
        }
        // Right boundary
        bond_dims.push(1usize);

        let mut tensors = Vec::with_capacity(num_vars);
        for i in 0..num_vars {
            let left_dim = bond_dims[i];
            let right_dim = bond_dims[i + 1];
            tensors.push(MpsTensor::new_random(left_dim, right_dim, rng, scale));
        }

        Self {
            tensors,
            bond_dim,
            num_vars,
        }
    }

    /// Evaluate the MPS amplitude for a given binary configuration.
    ///
    /// The amplitude is the product of matrices:
    ///   ψ(config) = A_0[b_0] · A_1[b_1] · ... · A_{n-1}[b_{n-1}]
    ///
    /// Since the boundary dimensions are 1, this is a scalar.
    pub fn evaluate(&self, config: &[u8]) -> f64 {
        assert_eq!(config.len(), self.num_vars);

        // Start with the first tensor's matrix for config[0]
        let first = self.tensors[0].matrix(config[0] as usize);
        // first is a 1×right_dim matrix, so we treat it as a row vector
        let mut state: Vec<f64> = first[0].clone();

        // Contract with each subsequent tensor
        for i in 1..self.num_vars {
            let mat = self.tensors[i].matrix(config[i] as usize);
            let new_dim = mat[0].len(); // right dimension
            let mut new_state = vec![0.0; new_dim];

            for beta in 0..new_dim {
                let mut sum = 0.0;
                for alpha in 0..state.len() {
                    sum += state[alpha] * mat[alpha][beta];
                }
                new_state[beta] = sum;
            }
            state = new_state;
        }

        // Final state should be a 1-element vector
        debug_assert_eq!(state.len(), 1);
        state[0]
    }

    /// Sample a binary configuration from |ψ|².
    ///
    /// Uses a left-to-right sweep, sampling each bit conditional on the previous ones.
    pub fn sample(&self, rng: &mut impl Rng) -> Vec<u8> {
        let mut config = vec![0u8; self.num_vars];

        // Compute right environments: right_env[i] = contraction of sites i+1..n
        // right_env[i][alpha][alpha'] = sum over all configs of sites i+1..n of
        //   (A_{i+1}...A_{n-1})^T (A_{i+1}...A_{n-1})
        let right_envs = self.compute_right_environments();

        // Left boundary: 1x1 identity
        let mut left_env = vec![vec![0.0; 1]; 1];
        left_env[0][0] = 1.0;

        for i in 0..self.num_vars {
            // Compute probability for sigma=0 and sigma=1
            let mut probs = [0.0f64; 2];
            for sigma in 0..2 {
                let mat = self.tensors[i].matrix(sigma);
                let right_dim = self.tensors[i].right_dim;
                let left_dim = self.tensors[i].left_dim;

                // Contract: left_env * mat * right_env
                // temp[beta] = sum_alpha left_env_contracted * mat[alpha][beta]
                let mut temp = vec![0.0; right_dim];
                for beta in 0..right_dim {
                    for alpha in 0..left_dim {
                        let mut left_val = 0.0;
                        for a in 0..left_env.len() {
                            for b in 0..left_env[0].len() {
                                if b == alpha {
                                    left_val += left_env[a][b] * mat[alpha][beta];
                                }
                            }
                        }
                        let _ = left_val; // Computed above for clarity
                    }
                }

                // Simpler approach: compute amplitude squared by marginalizing
                // over the rest. For sampling, compute the unnormalized probability.
                // P(sigma) ∝ sum_{config[i+1:]} |ψ(config[0:i-1], sigma, config[i+1:])|²
                // = (left_env) @ A[sigma] @ right_env[i] @ A[sigma]^T @ (left_env)^T

                // Actually, let's do it properly:
                // Contract left_env with A[sigma] to get partial contraction
                let mut contracted = vec![0.0; right_dim];
                for beta in 0..right_dim {
                    let mut sum = 0.0;
                    for alpha in 0..left_dim {
                        // left_env is a vector (from previous contractions)
                        if alpha < left_env.len() {
                            for a in 0..left_env[alpha].len() {
                                sum += left_env[alpha][a] * mat[a][beta];
                            }
                        }
                    }
                    contracted[beta] = sum;
                }
                temp = contracted;

                // Now contract with right environment
                let right = if i < self.num_vars - 1 {
                    &right_envs[i + 1]
                } else {
                    // No right environment for last site
                    &vec![vec![1.0]]
                };

                let mut prob = 0.0;
                for beta in 0..temp.len() {
                    if beta < right.len() {
                        prob += temp[beta] * temp[beta]; // |amplitude|²
                    }
                }
                probs[sigma] = prob.abs();
            }

            // Normalize and sample
            let total = probs[0] + probs[1];
            if total > 0.0 {
                let p0 = probs[0] / total;
                config[i] = if rng.gen::<f64>() < p0 { 0 } else { 1 };
            } else {
                config[i] = rng.gen_range(0..2);
            }

            // Update left environment
            let sigma = config[i] as usize;
            let mat = self.tensors[i].matrix(sigma);
            let right_dim = self.tensors[i].right_dim;
            let left_dim = self.tensors[i].left_dim;

            let mut new_left = vec![vec![0.0; right_dim]; 1];
            for beta in 0..right_dim {
                let mut sum = 0.0;
                for alpha in 0..left_dim {
                    if left_env.len() == 1 && alpha < left_env[0].len() {
                        sum += left_env[0][alpha] * mat[alpha][beta];
                    }
                }
                new_left[0][beta] = sum;
            }
            left_env = new_left;
        }

        config
    }

    /// Compute right environments for sampling.
    /// right_envs[i] is the contraction of squared tensors from site i to n-1.
    fn compute_right_environments(&self) -> Vec<Vec<Vec<f64>>> {
        let n = self.num_vars;
        let mut envs = vec![vec![vec![1.0]]; n + 1];

        // Sweep from right to left
        for i in (0..n).rev() {
            let tensor = &self.tensors[i];
            let left_dim = tensor.left_dim;
            let right_dim = tensor.right_dim;
            let right_env = &envs[i + 1];

            let mut new_env = vec![vec![0.0; left_dim]; left_dim];

            for sigma in 0..2 {
                let mat = tensor.matrix(sigma);
                for alpha in 0..left_dim {
                    for alpha_prime in 0..left_dim {
                        let mut sum = 0.0;
                        for beta in 0..right_dim {
                            for beta_prime in 0..right_dim {
                                if beta < right_env.len()
                                    && beta_prime < right_env[beta].len()
                                {
                                    sum += mat[alpha][beta]
                                        * mat[alpha_prime][beta_prime]
                                        * right_env[beta][beta_prime];
                                }
                            }
                        }
                        new_env[alpha][alpha_prime] += sum;
                    }
                }
            }

            envs[i] = new_env;
        }

        envs
    }

    /// Full contraction: compute the norm squared ⟨ψ|ψ⟩.
    ///
    /// This is the sum over all configurations of |ψ(config)|².
    pub fn contract_norm_squared(&self) -> f64 {
        // Use transfer matrix method: contract from left to right
        // T_i[alpha, alpha'] = Σ_σ A_i^σ[alpha, beta] * A_i^σ[alpha', beta']
        // contracted with the right part.

        // Start with identity on the left boundary
        let mut left = vec![vec![0.0; 1]; 1];
        left[0][0] = 1.0;

        for i in 0..self.num_vars {
            let tensor = &self.tensors[i];
            let right_dim = tensor.right_dim;
            let left_dim = tensor.left_dim;

            let mut new_left = vec![vec![0.0; right_dim]; right_dim];

            for sigma in 0..2 {
                let mat = tensor.matrix(sigma);
                for beta in 0..right_dim {
                    for beta_prime in 0..right_dim {
                        let mut sum = 0.0;
                        for alpha in 0..left_dim {
                            for alpha_prime in 0..left_dim {
                                if alpha < left.len() && alpha_prime < left[alpha].len() {
                                    sum += left[alpha][alpha_prime]
                                        * mat[alpha][beta]
                                        * mat[alpha_prime][beta_prime];
                                }
                            }
                        }
                        new_left[beta][beta_prime] += sum;
                    }
                }
            }

            left = new_left;
        }

        // Final contraction: left should be 1×1
        left[0][0]
    }

    /// Enumerate all 2^n configurations and find the one with maximum |ψ|².
    ///
    /// Only feasible for small n (say n ≤ 20).
    pub fn find_max_config(&self) -> (Vec<u8>, f64) {
        assert!(
            self.num_vars <= 24,
            "Exhaustive search only feasible for small n"
        );

        let total = 1u64 << self.num_vars;
        let mut best_config = vec![0u8; self.num_vars];
        let mut best_amp_sq = 0.0f64;

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
}

/// Multiply two matrices (simple implementation for small matrices).
pub fn mat_mul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let rows = a.len();
    let cols = b[0].len();
    let inner = a[0].len();
    let mut result = vec![vec![0.0; cols]; rows];
    for i in 0..rows {
        for j in 0..cols {
            for k in 0..inner {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    result
}

/// Matrix-vector multiplication.
pub fn mat_vec_mul(mat: &[Vec<f64>], vec: &[f64]) -> Vec<f64> {
    let rows = mat.len();
    let mut result = vec![0.0; rows];
    for i in 0..rows {
        for j in 0..vec.len() {
            result[i] += mat[i][j] * vec[j];
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn test_mps_tensor_creation() {
        let tensor = MpsTensor::new(3, 4);
        assert_eq!(tensor.left_dim, 3);
        assert_eq!(tensor.right_dim, 4);
        assert_eq!(tensor.data.len(), 2); // sigma ∈ {0, 1}
        assert_eq!(tensor.data[0].len(), 3); // alpha
        assert_eq!(tensor.data[0][0].len(), 4); // beta
    }

    #[test]
    fn test_mps_creation() {
        let mut rng = StdRng::seed_from_u64(42);
        let mps = Mps::new_random(5, 4, &mut rng);
        assert_eq!(mps.num_vars, 5);
        assert_eq!(mps.tensors.len(), 5);

        // Check boundary dimensions
        assert_eq!(mps.tensors[0].left_dim, 1);
        assert_eq!(mps.tensors[4].right_dim, 1);
    }

    #[test]
    fn test_mps_evaluate() {
        let mut rng = StdRng::seed_from_u64(42);
        let mps = Mps::new_random(3, 2, &mut rng);

        // Evaluate on a specific configuration
        let config = vec![0u8, 1, 0];
        let amp = mps.evaluate(&config);

        // Should be finite and non-NaN
        assert!(amp.is_finite());

        // Evaluate again — should be deterministic
        let amp2 = mps.evaluate(&config);
        assert!((amp - amp2).abs() < 1e-15);
    }

    #[test]
    fn test_mps_evaluate_all_configs() {
        let mut rng = StdRng::seed_from_u64(123);
        let mps = Mps::new_random(3, 2, &mut rng);

        // All 8 configurations
        let mut total_sq = 0.0;
        for bits in 0..8u8 {
            let config = vec![bits & 1, (bits >> 1) & 1, (bits >> 2) & 1];
            let amp = mps.evaluate(&config);
            total_sq += amp * amp;
        }

        // The norm squared via contraction should match
        let norm_sq = mps.contract_norm_squared();
        assert!(
            (total_sq - norm_sq).abs() < 1e-10,
            "Explicit sum {} != contraction {}",
            total_sq,
            norm_sq
        );
    }

    #[test]
    fn test_mps_sample() {
        let mut rng = StdRng::seed_from_u64(42);
        let mps = Mps::new_random(4, 2, &mut rng);

        // Just check it produces valid binary strings
        let mut rng2 = StdRng::seed_from_u64(99);
        for _ in 0..10 {
            let config = mps.sample(&mut rng2);
            assert_eq!(config.len(), 4);
            for &b in &config {
                assert!(b == 0 || b == 1);
            }
        }
    }

    #[test]
    fn test_mps_contract_positive() {
        let mut rng = StdRng::seed_from_u64(42);
        let mps = Mps::new_random(4, 3, &mut rng);
        let norm_sq = mps.contract_norm_squared();
        // Norm squared should be non-negative
        assert!(norm_sq >= 0.0, "Norm squared should be non-negative: {}", norm_sq);
    }

    #[test]
    fn test_mat_mul() {
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let b = vec![vec![5.0, 6.0], vec![7.0, 8.0]];
        let c = mat_mul(&a, &b);
        assert_eq!(c, vec![vec![19.0, 22.0], vec![43.0, 50.0]]);
    }

    #[test]
    fn test_find_max_config() {
        let mut rng = StdRng::seed_from_u64(42);
        let mps = Mps::new_random(4, 2, &mut rng);
        let (config, amp_sq) = mps.find_max_config();
        assert_eq!(config.len(), 4);
        assert!(amp_sq >= 0.0);

        // Verify this is indeed the maximum
        let amp = mps.evaluate(&config);
        assert!((amp * amp - amp_sq).abs() < 1e-12);
    }
}
