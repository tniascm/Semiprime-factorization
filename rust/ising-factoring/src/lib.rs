//! QUBO/Ising model formulation of integer factorization.
//!
//! Encodes N = p * q as a QUBO problem and solves via
//! simulated annealing with parallel tempering.
//!
//! Uses column-by-column carry-bit encoding:
//! For each output bit position k, the sum of partial products p_i * q_j
//! (where i+j=k) plus carry-in must equal bit k of N plus 2*carry-out.

use rand::Rng;
use rayon::prelude::*;

/// QUBO matrix representation (upper triangular).
/// H(x) = constant_offset + sum_i Q_ii * x_i + sum_{i<j} Q_ij * x_i * x_j
pub struct QuboMatrix {
    pub size: usize,
    pub coefficients: Vec<Vec<f64>>,
    /// Constant energy offset from expanding squared penalty terms.
    pub constant_offset: f64,
}

/// Variable layout for the QUBO factoring encoding.
struct VarLayout {
    p_bits: usize,
    q_bits: usize,
    /// z_{ij} = p_i * q_j auxiliary product variables.
    /// Index into the flat z array: i * q_bits + j
    z_offset: usize,
    /// For each column k, the carry variable indices (into the flat variable space).
    carry_vars: Vec<Vec<usize>>,
    total_vars: usize,
}

impl VarLayout {
    fn new(p_bits: usize, q_bits: usize) -> Self {
        let z_offset = p_bits + q_bits;
        let num_z = p_bits * q_bits;

        let num_columns = p_bits + q_bits; // output bit positions 0..num_columns-1
        let mut carry_vars: Vec<Vec<usize>> = Vec::with_capacity(num_columns);
        let mut next_var = z_offset + num_z;

        // For each column k, determine how many carry bits we need to represent
        // the carry-out. The maximum sum in column k is (number of partial products
        // in that column) + max carry-in. The carry-out is floor(sum/2), which
        // needs ceil(log2(max_sum)) bits to carry into next columns.
        // However, for simplicity and correctness, we use a single carry variable
        // per column: carry[k] represents the carry from column k to column k+1.
        // The carry can be multi-bit for large columns.
        for k in 0..num_columns {
            // Count partial products in column k
            let mut count = 0usize;
            for i in 0..p_bits {
                let j_needed = k as isize - i as isize;
                if j_needed >= 0 && (j_needed as usize) < q_bits {
                    count += 1;
                }
            }
            // Max possible sum in column k = count (from z's) + carry_in_max
            // For column 0, carry_in = 0, max sum = count
            // The carry out from column k can be at most floor((count + carry_in_max) / 2)
            // We need enough carry bits to represent this value.
            // Number of carry bits needed = ceil(log2(max_carry_out + 1))
            // For small numbers we compute iteratively.
            let max_carry_in = if k == 0 {
                0
            } else {
                // max carry from previous column
                let prev_carry_bits = carry_vars[k - 1].len();
                if prev_carry_bits == 0 {
                    0
                } else {
                    (1usize << prev_carry_bits) - 1
                }
            };
            let max_sum = count + max_carry_in;
            // carry_out = (sum - bit_k) / 2, max carry_out = floor(max_sum / 2)
            let max_carry_out = max_sum / 2;
            let num_carry_bits = if max_carry_out == 0 {
                0
            } else {
                (max_carry_out as f64).log2().floor() as usize + 1
            };

            let mut col_carries = Vec::with_capacity(num_carry_bits);
            for _ in 0..num_carry_bits {
                col_carries.push(next_var);
                next_var += 1;
            }
            carry_vars.push(col_carries);
        }

        VarLayout {
            p_bits,
            q_bits,
            z_offset,
            carry_vars,
            total_vars: next_var,
        }
    }

    fn p_var(&self, i: usize) -> usize {
        i
    }

    fn q_var(&self, j: usize) -> usize {
        self.p_bits + j
    }

    fn z_var(&self, i: usize, j: usize) -> usize {
        self.z_offset + i * self.q_bits + j
    }
}

/// Helper to add a value to Q[i][j] maintaining upper-triangular form.
fn add_qubo(coefficients: &mut [Vec<f64>], i: usize, j: usize, val: f64) {
    if i == j {
        coefficients[i][i] += val;
    } else if i < j {
        coefficients[i][j] += val;
    } else {
        coefficients[j][i] += val;
    }
}

/// Encode N = p * q as a QUBO problem.
/// Binary variables represent bits of p and q, plus auxiliary variables
/// for partial products (z_{ij} = p_i * q_j) and carry bits.
pub fn encode_factoring_qubo(n: u64, p_bits: usize, q_bits: usize) -> QuboMatrix {
    let layout = VarLayout::new(p_bits, q_bits);
    let total_vars = layout.total_vars;
    let mut coefficients = vec![vec![0.0; total_vars]; total_vars];
    let mut constant_offset = 0.0;

    // Penalty strength for enforcing z_{ij} = p_i * q_j
    let penalty = 8.0 * (n as f64);

    // === Part 1: Enforce z_{ij} = p_i * q_j ===
    // For each (i,j), add penalty: P * (z_{ij} - p_i * q_j)^2
    //   = P * (z_{ij}^2 - 2 * z_{ij} * p_i * q_j + p_i^2 * q_j^2)
    // But z, p, q are binary so z^2 = z, p^2 = p, q^2 = q, p*q is the product.
    // However p_i * q_j * z_{ij} is cubic. We use the standard penalty:
    //   P * (3 * z_{ij} + p_i * q_j - 2 * p_i * z_{ij} - 2 * q_j * z_{ij})
    // This is 0 when z_{ij} = p_i * q_j and positive otherwise.
    for i in 0..p_bits {
        for j in 0..q_bits {
            let pi = layout.p_var(i);
            let qj = layout.q_var(j);
            let zij = layout.z_var(i, j);

            add_qubo(&mut coefficients, zij, zij, 3.0 * penalty);
            add_qubo(&mut coefficients, pi, qj, 1.0 * penalty);
            add_qubo(&mut coefficients, pi, zij, -2.0 * penalty);
            add_qubo(&mut coefficients, qj, zij, -2.0 * penalty);
        }
    }

    // === Part 2: Column constraints ===
    // For each output bit column k:
    //   sum of z_{ij} where i+j=k + carry_in = n_k + 2 * carry_out
    // where n_k is bit k of N.
    // We encode carry_out as a binary number using carry variables.
    //
    // Rewrite as: (sum_of_z + carry_in_value - n_k - 2 * carry_out_value)^2 = 0
    // and add this as a penalty.

    let num_columns = p_bits + q_bits;

    for k in 0..num_columns {
        let n_k = if k < 64 { ((n >> k) & 1) as f64 } else { 0.0 };

        // Collect z variables contributing to this column
        let mut z_vars_in_col: Vec<usize> = Vec::new();
        for i in 0..p_bits {
            let j_needed = k as isize - i as isize;
            if j_needed >= 0 && (j_needed as usize) < q_bits {
                z_vars_in_col.push(layout.z_var(i, j_needed as usize));
            }
        }

        // Carry-in variables (from column k-1)
        let carry_in_vars: Vec<usize> = if k > 0 {
            layout.carry_vars[k - 1].clone()
        } else {
            Vec::new()
        };

        // Carry-out variables (for this column k)
        let carry_out_vars: &[usize] = &layout.carry_vars[k];

        // We want: (S - n_k)^2 = 0 where
        //   S = sum(z_vars) + sum(2^b * carry_in_b) - sum(2^(b+1) * carry_out_b)
        // but carry_out contributes to next column, so:
        //   S = sum(z_vars) + carry_in_value - n_k - 2 * carry_out_value
        // Define the penalty: (S)^2 where S = sum(z) + carry_in_val - n_k - 2*carry_out_val
        //
        // Collect all terms with their coefficients:
        // Each z variable has coefficient +1
        // Each carry_in bit b has coefficient +2^b
        // Each carry_out bit b has coefficient -2^(b+1)  (the factor 2 is because
        //   carry_out represents bits shifted left by 1)
        // Constant term: -n_k

        struct Term {
            var: usize,
            coeff: f64,
        }

        let mut terms: Vec<Term> = Vec::new();

        for &zv in &z_vars_in_col {
            terms.push(Term { var: zv, coeff: 1.0 });
        }

        for (b, &cv) in carry_in_vars.iter().enumerate() {
            terms.push(Term { var: cv, coeff: (1u64 << b) as f64 });
        }

        for (b, &cv) in carry_out_vars.iter().enumerate() {
            terms.push(Term { var: cv, coeff: -((2u64 << b) as f64) });
        }

        let constant = -n_k;

        // Expand (sum_t coeff_t * x_t + constant)^2 as a QUBO penalty
        let col_penalty = penalty;

        // constant^2 term (global offset)
        constant_offset += col_penalty * constant * constant;

        // Linear terms: 2 * constant * coeff_t * x_t
        // and diagonal from coeff_t^2 * x_t^2 = coeff_t^2 * x_t (since x_t is binary)
        for t in &terms {
            let linear = t.coeff * t.coeff + 2.0 * constant * t.coeff;
            add_qubo(&mut coefficients, t.var, t.var, col_penalty * linear);
        }

        // Cross terms: 2 * coeff_s * coeff_t * x_s * x_t
        for s in 0..terms.len() {
            for t in (s + 1)..terms.len() {
                let cross = 2.0 * terms[s].coeff * terms[t].coeff;
                add_qubo(&mut coefficients, terms[s].var, terms[t].var, col_penalty * cross);
            }
        }
    }

    QuboMatrix {
        size: total_vars,
        coefficients,
        constant_offset,
    }
}

/// Evaluate QUBO energy for a given binary assignment.
pub fn evaluate_qubo(qubo: &QuboMatrix, assignment: &[bool]) -> f64 {
    let mut energy = qubo.constant_offset;
    for i in 0..qubo.size {
        if assignment[i] {
            energy += qubo.coefficients[i][i];
            for j in (i + 1)..qubo.size {
                if assignment[j] {
                    energy += qubo.coefficients[i][j];
                }
            }
        }
    }
    energy
}

/// Simulated annealing solver for QUBO.
pub fn simulated_annealing(
    qubo: &QuboMatrix,
    initial_temp: f64,
    final_temp: f64,
    steps: usize,
) -> (Vec<bool>, f64) {
    let mut rng = rand::thread_rng();
    let mut state: Vec<bool> = (0..qubo.size).map(|_| rng.gen()).collect();
    let mut energy = evaluate_qubo(qubo, &state);
    let mut best_state = state.clone();
    let mut best_energy = energy;

    let cooling_rate = (final_temp / initial_temp).powf(1.0 / steps as f64);
    let mut temp = initial_temp;

    for _ in 0..steps {
        // Flip a random bit
        let flip_idx = rng.gen_range(0..qubo.size);

        // Compute delta energy efficiently
        let delta = compute_flip_delta(qubo, &state, flip_idx);

        if delta < 0.0 || rng.gen::<f64>() < (-delta / temp).exp() {
            state[flip_idx] = !state[flip_idx];
            energy += delta;
            if energy < best_energy {
                best_energy = energy;
                best_state = state.clone();
            }
        }

        temp *= cooling_rate;
    }

    (best_state, best_energy)
}

/// Compute the change in energy from flipping bit `flip_idx`.
fn compute_flip_delta(qubo: &QuboMatrix, state: &[bool], flip_idx: usize) -> f64 {
    let mut delta = 0.0;
    let currently_set = state[flip_idx];

    if currently_set {
        // Turning off: subtract contributions
        delta -= qubo.coefficients[flip_idx][flip_idx];
        for j in 0..qubo.size {
            if j != flip_idx && state[j] {
                if flip_idx < j {
                    delta -= qubo.coefficients[flip_idx][j];
                } else {
                    delta -= qubo.coefficients[j][flip_idx];
                }
            }
        }
    } else {
        // Turning on: add contributions
        delta += qubo.coefficients[flip_idx][flip_idx];
        for j in 0..qubo.size {
            if j != flip_idx && state[j] {
                if flip_idx < j {
                    delta += qubo.coefficients[flip_idx][j];
                } else {
                    delta += qubo.coefficients[j][flip_idx];
                }
            }
        }
    }

    delta
}

/// Parallel tempering: run multiple SA instances at different temperatures.
pub fn parallel_tempering(
    qubo: &QuboMatrix,
    num_replicas: usize,
    steps_per_replica: usize,
) -> (Vec<bool>, f64) {
    let temps: Vec<f64> = (0..num_replicas)
        .map(|i| 10.0_f64.powf(2.0 - 4.0 * i as f64 / (num_replicas - 1) as f64))
        .collect();

    let results: Vec<(Vec<bool>, f64)> = temps
        .par_iter()
        .map(|&temp| simulated_annealing(qubo, temp, temp * 0.001, steps_per_replica))
        .collect();

    results
        .into_iter()
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .unwrap()
}

/// Extract p and q from QUBO solution.
/// Only reads the first p_bits and next q_bits variables; ignores auxiliary vars.
pub fn extract_factors(solution: &[bool], p_bits: usize, q_bits: usize) -> (u64, u64) {
    let mut p = 0u64;
    let mut q = 0u64;

    for i in 0..p_bits {
        if solution[i] {
            p |= 1 << i;
        }
    }
    for j in 0..q_bits {
        if solution[p_bits + j] {
            q |= 1 << j;
        }
    }

    (p, q)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qubo_encoding() {
        let qubo = encode_factoring_qubo(15, 4, 4);
        // Size now includes auxiliary z and carry variables
        assert!(qubo.size >= 8);
        // Should have p_bits + q_bits + p_bits*q_bits z vars + carry vars
        assert!(qubo.size > 8);
    }

    #[test]
    fn test_simulated_annealing_runs() {
        let qubo = encode_factoring_qubo(15, 4, 4);
        let (solution, energy) = simulated_annealing(&qubo, 100.0, 0.01, 10_000);
        assert_eq!(solution.len(), qubo.size);
        // Energy should be finite
        assert!(energy.is_finite());
    }

    #[test]
    fn test_correct_factorization_has_zero_energy() {
        // Test that the correct assignment for 15 = 3 * 5 yields zero energy.
        // p = 3 = 0b0011, q = 5 = 0b0101 (or p=5, q=3)
        let p_bits = 4;
        let q_bits = 4;
        let n: u64 = 15;
        let qubo = encode_factoring_qubo(n, p_bits, q_bits);

        // Try p=3, q=5
        let p: u64 = 3;
        let q: u64 = 5;

        let mut assignment = vec![false; qubo.size];

        // Set p bits
        for i in 0..p_bits {
            assignment[i] = ((p >> i) & 1) == 1;
        }
        // Set q bits
        for j in 0..q_bits {
            assignment[p_bits + j] = ((q >> j) & 1) == 1;
        }
        // Set z_{ij} = p_i * q_j
        let layout = VarLayout::new(p_bits, q_bits);
        for i in 0..p_bits {
            for j in 0..q_bits {
                let pi = ((p >> i) & 1) == 1;
                let qj = ((q >> j) & 1) == 1;
                assignment[layout.z_var(i, j)] = pi && qj;
            }
        }
        // Set carry variables correctly
        // For each column, compute the sum and determine carries
        let num_columns = p_bits + q_bits;
        let mut carry_in: u64 = 0;
        for k in 0..num_columns {
            let n_k = if k < 64 { ((n >> k) & 1) as u64 } else { 0 };

            // Sum of z_{ij} where i+j = k
            let mut col_sum: u64 = 0;
            for i in 0..p_bits {
                let j_needed = k as isize - i as isize;
                if j_needed >= 0 && (j_needed as usize) < q_bits {
                    let pi = ((p >> i) & 1) == 1;
                    let qj = ((q >> (j_needed as usize)) & 1) == 1;
                    if pi && qj {
                        col_sum += 1;
                    }
                }
            }
            col_sum += carry_in;

            // col_sum should equal n_k + 2 * carry_out
            // carry_out = (col_sum - n_k) / 2
            let carry_out = (col_sum - n_k) / 2;

            // Set carry-out bits for this column
            for (b, &cv) in layout.carry_vars[k].iter().enumerate() {
                assignment[cv] = ((carry_out >> b) & 1) == 1;
            }

            // Carry-in for next column
            carry_in = carry_out;
        }

        let energy = evaluate_qubo(&qubo, &assignment);
        assert!(
            energy.abs() < 1e-6,
            "Correct factorization should have ~zero energy, got {}",
            energy
        );
    }

    #[test]
    fn test_qubo_factors_15() {
        // Factor 15 = 3 * 5
        let p_bits = 4;
        let q_bits = 4;
        let n: u64 = 15;
        let qubo = encode_factoring_qubo(n, p_bits, q_bits);

        // Run multiple attempts to increase chance of finding factors
        // (simulated annealing is probabilistic, so we retry generously)
        let mut found = false;
        for _ in 0..30 {
            let (solution, _energy) = parallel_tempering(&qubo, 16, 200_000);
            let (p, q) = extract_factors(&solution, p_bits, q_bits);
            if p > 1 && q > 1 && p * q == n {
                found = true;
                break;
            }
        }
        assert!(found, "Failed to factor 15 = 3 * 5");
    }

    #[test]
    fn test_qubo_factors_77() {
        // Factor 77 = 7 * 11
        let p_bits = 4;
        let q_bits = 4;
        let n: u64 = 77;
        let qubo = encode_factoring_qubo(n, p_bits, q_bits);

        let mut found = false;
        for _ in 0..20 {
            let (solution, _energy) = parallel_tempering(&qubo, 16, 200_000);
            let (p, q) = extract_factors(&solution, p_bits, q_bits);
            if p > 1 && q > 1 && p * q == n {
                found = true;
                break;
            }
        }
        assert!(found, "Failed to factor 77 = 7 * 11");
    }
}
