//! DMRG-style variational optimization for the MPS.
//!
//! The cost function ||Lx - t||² is converted to a quadratic binary optimization
//! problem (QUBO) over the binary variables b_{i,j} that encode the lattice exponents.
//! We then minimize ⟨ψ|H|ψ⟩ / ⟨ψ|ψ⟩ using alternating least squares (ALS) sweeps,
//! which is the tensor network analog of DMRG.

use crate::tensor::Mps;

/// A quadratic term c * b_i * b_j in the cost Hamiltonian.
/// When i == j, this represents a linear term c * b_i (since b_i² = b_i for binary).
#[derive(Debug, Clone)]
pub struct QuadraticTerm {
    /// First variable index.
    pub i: usize,
    /// Second variable index (can equal i for linear terms).
    pub j: usize,
    /// Coefficient.
    pub coeff: f64,
}

/// Cost Hamiltonian H = Σ c_{ij} b_i b_j + constant.
///
/// For binary variables b ∈ {0,1}: b² = b, so diagonal quadratic terms
/// are the same as linear terms.
#[derive(Debug, Clone)]
pub struct CostHamiltonian {
    /// Quadratic and linear terms.
    pub terms: Vec<QuadraticTerm>,
    /// Constant offset.
    pub constant: f64,
    /// Total number of binary variables.
    pub num_vars: usize,
}

impl CostHamiltonian {
    /// Evaluate the cost function for a given binary configuration.
    pub fn evaluate(&self, config: &[u8]) -> f64 {
        let mut cost = self.constant;
        for term in &self.terms {
            let bi = config[term.i] as f64;
            let bj = config[term.j] as f64;
            cost += term.coeff * bi * bj;
        }
        cost
    }
}

/// Build the cost Hamiltonian from the Schnorr lattice.
///
/// The cost function is ||Lx||² where x = (e_1, ..., e_k, 0) and each e_i
/// is encoded in binary: e_i = Σ_j b_{i,j} * 2^j - exponent_bound.
///
/// The lattice multiplication Lx gives a vector, and ||Lx||² expands to
/// a polynomial in the binary variables b_{i,j}.
///
/// # Arguments
/// - `lattice`: The (k+1)×(k+1) Schnorr lattice matrix
/// - `target`: Target vector (usually zero for shortest vector)
/// - `num_exponents`: k, the number of factor base primes
/// - `bits_per_exp`: B, number of bits to encode each exponent
/// - `exponent_bound`: E, so that e_i = binary_value - E
pub fn build_cost_from_lattice(
    lattice: &[Vec<i64>],
    target: &[i64],
    num_exponents: usize,
    bits_per_exp: usize,
    exponent_bound: i64,
) -> CostHamiltonian {
    let num_vars = num_exponents * bits_per_exp;
    let dim = lattice.len();

    // The exponent vector x has entries:
    //   x_i = Σ_{j=0}^{B-1} b_{i*B + j} * 2^j - E   for i = 0..k-1
    //   x_k = 0 (we don't optimize the last coordinate)
    //
    // Cost = ||L * x_full - target||² where x_full = (x_0, ..., x_{k-1}, 0)
    //
    // = Σ_r (Σ_i L[r][i] * x_i - target[r])²
    //
    // Expanding x_i in terms of binary variables:
    //   x_i = Σ_j b_{i*B+j} * 2^j - E
    //
    // So: Σ_i L[r][i] * x_i = Σ_i L[r][i] * (Σ_j b_{i*B+j} * 2^j - E)
    //   = Σ_i Σ_j L[r][i] * 2^j * b_{i*B+j} - E * Σ_i L[r][i]
    //
    // Let c_{r, i*B+j} = L[r][i] * 2^j  (coefficient of b_{i*B+j} in row r)
    // Let d_r = -E * Σ_i L[r][i] - target[r]  (constant for row r)
    //
    // Then: row_r = Σ_v c_{r,v} * b_v + d_r
    // Cost = Σ_r row_r² = Σ_r (Σ_v c_{r,v} * b_v + d_r)²

    // Precompute the linear coefficients for each row and binary variable
    let mut row_coeffs: Vec<Vec<f64>> = Vec::with_capacity(dim);
    let mut row_constants: Vec<f64> = Vec::with_capacity(dim);

    for r in 0..dim {
        let mut coeffs = vec![0.0; num_vars];
        let mut constant = 0.0;

        for i in 0..num_exponents {
            let l_ri = lattice[r][i] as f64;
            // Contribution from offset: -E * L[r][i]
            constant -= exponent_bound as f64 * l_ri;
            // Contribution from binary variables
            for j in 0..bits_per_exp {
                let var_idx = i * bits_per_exp + j;
                coeffs[var_idx] = l_ri * (1i64 << j) as f64;
            }
        }

        // Subtract target
        if r < target.len() {
            constant -= target[r] as f64;
        }

        row_coeffs.push(coeffs);
        row_constants.push(constant);
    }

    // Now expand Cost = Σ_r (Σ_v c_{r,v} * b_v + d_r)²
    //                = Σ_r [ Σ_{v,w} c_{r,v} * c_{r,w} * b_v * b_w
    //                      + 2 * d_r * Σ_v c_{r,v} * b_v
    //                      + d_r² ]

    let mut terms: Vec<QuadraticTerm> = Vec::new();
    let mut constant = 0.0;

    // Use a map to aggregate coefficients for each (i,j) pair
    let mut coeff_map: Vec<Vec<f64>> = vec![vec![0.0; num_vars]; num_vars];
    let mut linear_map: Vec<f64> = vec![0.0; num_vars];

    for r in 0..dim {
        let coeffs = &row_coeffs[r];
        let d = row_constants[r];

        // Quadratic: c_{r,v} * c_{r,w} for all v, w
        for v in 0..num_vars {
            if coeffs[v] == 0.0 {
                continue;
            }
            for w in v..num_vars {
                if coeffs[w] == 0.0 {
                    continue;
                }
                let c = if v == w {
                    coeffs[v] * coeffs[w]
                } else {
                    2.0 * coeffs[v] * coeffs[w]
                };
                coeff_map[v][w] += c;
            }
        }

        // Linear: 2 * d_r * c_{r,v}
        for v in 0..num_vars {
            linear_map[v] += 2.0 * d * coeffs[v];
        }

        // Constant: d_r²
        constant += d * d;
    }

    // Convert to QuadraticTerm list
    // For binary variables: b_v² = b_v, so diagonal quadratic terms become linear
    for v in 0..num_vars {
        // Diagonal quadratic + linear -> combined linear
        let combined_linear = coeff_map[v][v] + linear_map[v];
        if combined_linear.abs() > 1e-15 {
            terms.push(QuadraticTerm {
                i: v,
                j: v,
                coeff: combined_linear,
            });
        }

        // Off-diagonal quadratic
        for w in (v + 1)..num_vars {
            if coeff_map[v][w].abs() > 1e-15 {
                terms.push(QuadraticTerm {
                    i: v,
                    j: w,
                    coeff: coeff_map[v][w],
                });
            }
        }
    }

    CostHamiltonian {
        terms,
        constant,
        num_vars,
    }
}

/// Perform DMRG-style sweeps to optimize the MPS to minimize ⟨ψ|H|ψ⟩/⟨ψ|ψ⟩.
///
/// Uses a simplified single-site DMRG approach:
/// 1. Sweep left-to-right, then right-to-left
/// 2. At each site, compute the effective 2×2 Hamiltonian
/// 3. Find the ground state and update the tensor
///
/// Returns the final energy (cost function value).
pub fn optimize_sweep(mps: &mut Mps, hamiltonian: &CostHamiltonian, num_sweeps: usize) -> f64 {
    let n = mps.num_vars;
    let mut best_energy = f64::MAX;

    for _sweep in 0..num_sweeps {
        // Left-to-right sweep
        for site in 0..n {
            optimize_single_site(mps, hamiltonian, site);
        }

        // Right-to-left sweep
        for site in (0..n).rev() {
            optimize_single_site(mps, hamiltonian, site);
        }

        // Compute current energy
        let energy = compute_expectation(mps, hamiltonian);
        if energy < best_energy {
            best_energy = energy;
        }
    }

    best_energy
}

/// Optimize a single site's tensor in the MPS.
///
/// For the given site, we compute the effective Hamiltonian by contracting
/// the rest of the MPS network, then find the optimal 2-element tensor
/// (for sigma=0 and sigma=1).
fn optimize_single_site(mps: &mut Mps, hamiltonian: &CostHamiltonian, site: usize) {
    let left_dim = mps.tensors[site].left_dim;
    let right_dim = mps.tensors[site].right_dim;

    // Compute left and right environments
    let left_env = compute_left_env(mps, site);
    let right_env = compute_right_env(mps, site);

    // For each sigma value, compute the optimal matrix entries
    // We use a gradient-based local update:
    // The effective Hamiltonian for this site is a matrix acting on
    // the combined (sigma, alpha, beta) index space.

    // Simple approach: try both sigma=0 and sigma=1, and for each,
    // evaluate the cost and choose the configuration that minimizes energy.

    // More sophisticated: compute the effective Hamiltonian matrix and diagonalize.
    // For our simplified approach, we'll use a variational update based on
    // evaluating the cost function contribution.

    // Compute effective cost matrix for this site's tensor
    // h_eff[sigma][alpha][beta][sigma'][alpha'][beta'] but this is expensive.
    // Instead, use a simpler approach: perturb each element and compute gradient.

    let step_size = 0.01;
    let num_steps = 5;

    for _ in 0..num_steps {
        for sigma in 0..2 {
            for alpha in 0..left_dim {
                for beta in 0..right_dim {
                    // Compute gradient by finite difference
                    let original = mps.tensors[site].data[sigma][alpha][beta];

                    // Forward
                    mps.tensors[site].data[sigma][alpha][beta] = original + step_size;
                    let e_plus = compute_local_energy(mps, hamiltonian, &left_env, &right_env, site);

                    // Backward
                    mps.tensors[site].data[sigma][alpha][beta] = original - step_size;
                    let e_minus =
                        compute_local_energy(mps, hamiltonian, &left_env, &right_env, site);

                    // Gradient
                    let grad = (e_plus - e_minus) / (2.0 * step_size);

                    // Update
                    mps.tensors[site].data[sigma][alpha][beta] = original - step_size * grad;
                }
            }
        }
    }
}

/// Compute the left environment: contraction of sites 0..site-1.
/// Returns a matrix of dimension [left_bond × left_bond] representing
/// the partial contraction from the left boundary up to (but not including)
/// the given site.
fn compute_left_env(mps: &Mps, site: usize) -> Vec<Vec<f64>> {
    // Start with 1×1 identity
    let mut env = vec![vec![1.0f64]];

    for i in 0..site {
        let tensor = &mps.tensors[i];
        let right_dim = tensor.right_dim;
        let left_dim = tensor.left_dim;
        let mut new_env = vec![vec![0.0; right_dim]; right_dim];

        for sigma in 0..2 {
            let mat = tensor.matrix(sigma);
            for beta in 0..right_dim {
                for beta_prime in 0..right_dim {
                    let mut sum = 0.0;
                    for alpha in 0..left_dim {
                        for alpha_prime in 0..left_dim {
                            if alpha < env.len() && alpha_prime < env[alpha].len() {
                                sum += env[alpha][alpha_prime]
                                    * mat[alpha][beta]
                                    * mat[alpha_prime][beta_prime];
                            }
                        }
                    }
                    new_env[beta][beta_prime] += sum;
                }
            }
        }

        env = new_env;
    }

    env
}

/// Compute the right environment: contraction of sites site+1..n-1.
fn compute_right_env(mps: &Mps, site: usize) -> Vec<Vec<f64>> {
    let n = mps.num_vars;
    let mut env = vec![vec![1.0f64]];

    for i in (site + 1..n).rev() {
        let tensor = &mps.tensors[i];
        let left_dim = tensor.left_dim;
        let right_dim = tensor.right_dim;
        let mut new_env = vec![vec![0.0; left_dim]; left_dim];

        for sigma in 0..2 {
            let mat = tensor.matrix(sigma);
            for alpha in 0..left_dim {
                for alpha_prime in 0..left_dim {
                    let mut sum = 0.0;
                    for beta in 0..right_dim {
                        for beta_prime in 0..right_dim {
                            if beta < env.len() && beta_prime < env[beta].len() {
                                sum += mat[alpha][beta]
                                    * mat[alpha_prime][beta_prime]
                                    * env[beta][beta_prime];
                            }
                        }
                    }
                    new_env[alpha][alpha_prime] += sum;
                }
            }
        }

        env = new_env;
    }

    env
}

/// Compute the local energy contribution involving the specified site.
///
/// This is a simplified computation that evaluates ⟨ψ|H|ψ⟩/⟨ψ|ψ⟩ using
/// the environments and the current site tensor.
fn compute_local_energy(
    mps: &Mps,
    hamiltonian: &CostHamiltonian,
    _left_env: &[Vec<f64>],
    _right_env: &[Vec<f64>],
    _site: usize,
) -> f64 {
    // For the simplified implementation, we evaluate the expectation value
    // by sampling. This is approximate but avoids the full environment contraction.
    compute_expectation(mps, hamiltonian)
}

/// Compute ⟨ψ|H|ψ⟩ / ⟨ψ|ψ⟩ — the expectation value of the cost Hamiltonian.
///
/// For small systems, we enumerate all configurations.
/// For larger systems, this uses sampling.
pub fn compute_expectation(mps: &Mps, hamiltonian: &CostHamiltonian) -> f64 {
    let n = mps.num_vars;

    if n <= 20 {
        // Exact enumeration
        let total_configs = 1u64 << n;
        let mut energy_sum = 0.0;
        let mut norm_sum = 0.0;

        for bits in 0..total_configs {
            let config: Vec<u8> = (0..n).map(|j| ((bits >> j) & 1) as u8).collect();
            let amp = mps.evaluate(&config);
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
        // Sampling-based estimation
        let mut rng = rand::thread_rng();
        let num_samples = 1000;
        let mut energy_sum = 0.0;
        let mut weight_sum = 0.0;

        for _ in 0..num_samples {
            let config = mps.sample(&mut rng);
            let amp = mps.evaluate(&config);
            let amp_sq = amp * amp;
            let cost = hamiltonian.evaluate(&config);
            energy_sum += amp_sq * cost;
            weight_sum += amp_sq;
        }

        if weight_sum.abs() < 1e-30 {
            return f64::MAX;
        }

        energy_sum / weight_sum
    }
}

/// Extract the integer exponent solution from the optimized MPS.
///
/// Finds the configuration with maximum amplitude and converts the
/// binary encoding back to integer exponents.
pub fn extract_solution(
    mps: &Mps,
    num_exponents: usize,
    bits_per_exp: usize,
    exponent_bound: i64,
) -> Vec<i64> {
    let n = mps.num_vars;

    // Find the best configuration
    let (config, _) = if n <= 24 {
        mps.find_max_config()
    } else {
        // For larger systems, sample many configurations and take the best
        let mut rng = rand::thread_rng();
        let mut best_config = mps.sample(&mut rng);
        let mut best_amp = mps.evaluate(&best_config).abs();

        for _ in 0..10000 {
            let config = mps.sample(&mut rng);
            let amp = mps.evaluate(&config).abs();
            if amp > best_amp {
                best_amp = amp;
                best_config = config;
            }
        }
        (best_config, best_amp * best_amp)
    };

    // Decode binary configuration to integer exponents
    binary_to_exponents(&config, num_exponents, bits_per_exp, exponent_bound)
}

/// Convert a binary configuration to integer exponents.
///
/// Each exponent e_i = Σ_{j=0}^{B-1} b_{i*B+j} * 2^j - exponent_bound
pub fn binary_to_exponents(
    config: &[u8],
    num_exponents: usize,
    bits_per_exp: usize,
    exponent_bound: i64,
) -> Vec<i64> {
    let mut exponents = Vec::with_capacity(num_exponents);

    for i in 0..num_exponents {
        let mut value: i64 = 0;
        for j in 0..bits_per_exp {
            let idx = i * bits_per_exp + j;
            if idx < config.len() && config[idx] == 1 {
                value += 1i64 << j;
            }
        }
        exponents.push(value - exponent_bound);
    }

    exponents
}

/// Convert integer exponents to a binary configuration.
///
/// Inverse of `binary_to_exponents`.
pub fn exponents_to_binary(
    exponents: &[i64],
    bits_per_exp: usize,
    exponent_bound: i64,
) -> Vec<u8> {
    let num_vars = exponents.len() * bits_per_exp;
    let mut config = vec![0u8; num_vars];

    for (i, &e) in exponents.iter().enumerate() {
        let value = (e + exponent_bound) as u64;
        for j in 0..bits_per_exp {
            if (value >> j) & 1 == 1 {
                config[i * bits_per_exp + j] = 1;
            }
        }
    }

    config
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn test_binary_to_exponents_roundtrip() {
        let exponents = vec![2i64, -1, 0, 3];
        let bits = 4;
        let bound = 7;

        let binary = exponents_to_binary(&exponents, bits, bound);
        let recovered = binary_to_exponents(&binary, exponents.len(), bits, bound);

        assert_eq!(exponents, recovered);
    }

    #[test]
    fn test_binary_encoding_bounds() {
        let bits = 3;
        let bound = 3;

        // Minimum: all zeros -> 0 - 3 = -3
        let config = vec![0u8; 3];
        let exp = binary_to_exponents(&config, 1, bits, bound);
        assert_eq!(exp[0], -3);

        // Maximum: all ones -> 7 - 3 = 4
        let config = vec![1u8; 3];
        let exp = binary_to_exponents(&config, 1, bits, bound);
        assert_eq!(exp[0], 4);
    }

    #[test]
    fn test_cost_hamiltonian_evaluate() {
        // Simple 2x2 lattice, cost = (x0 + x1)²
        let hamiltonian = CostHamiltonian {
            terms: vec![
                QuadraticTerm {
                    i: 0,
                    j: 0,
                    coeff: 1.0,
                },
                QuadraticTerm {
                    i: 0,
                    j: 1,
                    coeff: 2.0,
                },
                QuadraticTerm {
                    i: 1,
                    j: 1,
                    coeff: 1.0,
                },
            ],
            constant: 0.0,
            num_vars: 2,
        };

        // config [0, 0]: cost = 0
        assert!((hamiltonian.evaluate(&[0, 0]) - 0.0).abs() < 1e-10);
        // config [1, 0]: cost = 1
        assert!((hamiltonian.evaluate(&[1, 0]) - 1.0).abs() < 1e-10);
        // config [0, 1]: cost = 1
        assert!((hamiltonian.evaluate(&[0, 1]) - 1.0).abs() < 1e-10);
        // config [1, 1]: cost = 1 + 2 + 1 = 4
        assert!((hamiltonian.evaluate(&[1, 1]) - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_build_cost_simple() {
        // 1D lattice: L = [[1, a], [0, b]]
        // For num_exponents=1, bits_per_exp=1, exponent_bound=0:
        //   x_0 = b_0 - 0 = b_0
        //   Cost = (1 * x_0)² + (0 * x_0 - 0)² = x_0² = b_0
        let lattice = vec![vec![1, 10], vec![0, 20]];
        let target = vec![0, 0];

        let h = build_cost_from_lattice(&lattice, &target, 1, 1, 0);

        // b_0 = 0: cost should be 0
        let cost_0 = h.evaluate(&[0]);
        // b_0 = 1: x_0 = 1, cost = (1*1)^2 = 1 (from row 0)
        let cost_1 = h.evaluate(&[1]);

        assert!(cost_0 < cost_1, "Zero config should have lower cost");
    }

    #[test]
    fn test_optimize_reduces_energy() {
        let mut rng = StdRng::seed_from_u64(42);

        // Simple 2-variable problem
        let hamiltonian = CostHamiltonian {
            terms: vec![
                QuadraticTerm {
                    i: 0,
                    j: 0,
                    coeff: 1.0,
                },
                QuadraticTerm {
                    i: 1,
                    j: 1,
                    coeff: 1.0,
                },
                QuadraticTerm {
                    i: 0,
                    j: 1,
                    coeff: 2.0,
                },
            ],
            constant: 0.0,
            num_vars: 2,
        };

        let mut mps = Mps::new_random(2, 2, &mut rng);
        let initial_energy = compute_expectation(&mps, &hamiltonian);
        let final_energy = optimize_sweep(&mut mps, &hamiltonian, 5);

        // Energy should decrease or stay the same
        assert!(
            final_energy <= initial_energy + 1e-6,
            "Energy should decrease: {} -> {}",
            initial_energy,
            final_energy
        );
    }

    #[test]
    fn test_extract_solution_small() {
        let mut rng = StdRng::seed_from_u64(42);
        let mps = Mps::new_random(6, 2, &mut rng);

        // 2 exponents, 3 bits each, bound = 3
        let exponents = extract_solution(&mps, 2, 3, 3);
        assert_eq!(exponents.len(), 2);

        // Each exponent should be in [-3, 4]
        for &e in &exponents {
            assert!(e >= -3 && e <= 4, "Exponent {} out of range", e);
        }
    }
}
