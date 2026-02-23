//! Symbolic regression for discovering parameter formulas.
//!
//! Uses a small GP system to find mathematical formulas that predict
//! optimal parameter values from input bit size. For example, it might
//! discover that "B1_optimal ≈ 3.2 × bits^(2/3)" for ECM macro blocks.
//!
//! The approach:
//! 1. Collect empirical data: (bit_size, param_value, success_rate) tuples
//! 2. Evolve formula trees over {+, -, ×, ÷, sqrt, log, pow, const}
//! 3. Score by R² on held-out bit sizes
//! 4. Return best formulas as parameter-selection rules

use rand::Rng;
use std::fmt;

// ---------------------------------------------------------------------------
// Formula tree (symbolic expression)
// ---------------------------------------------------------------------------

/// A node in a symbolic expression tree.
#[derive(Debug, Clone)]
pub enum FormulaNode {
    /// A constant value.
    Const(f64),
    /// The input variable (typically bit_size as f64).
    Var,
    /// Addition: left + right.
    Add(Box<FormulaNode>, Box<FormulaNode>),
    /// Subtraction: left - right.
    Sub(Box<FormulaNode>, Box<FormulaNode>),
    /// Multiplication: left × right.
    Mul(Box<FormulaNode>, Box<FormulaNode>),
    /// Protected division: left / right (returns 1.0 if right ≈ 0).
    Div(Box<FormulaNode>, Box<FormulaNode>),
    /// Square root: sqrt(|child|).
    Sqrt(Box<FormulaNode>),
    /// Natural logarithm: ln(|child| + 1).
    Log(Box<FormulaNode>),
    /// Power: base^exponent (clamped to avoid overflow).
    Pow(Box<FormulaNode>, Box<FormulaNode>),
}

impl FormulaNode {
    /// Evaluate the formula tree on the given input value.
    pub fn evaluate(&self, x: f64) -> f64 {
        match self {
            FormulaNode::Const(c) => *c,
            FormulaNode::Var => x,
            FormulaNode::Add(a, b) => a.evaluate(x) + b.evaluate(x),
            FormulaNode::Sub(a, b) => a.evaluate(x) - b.evaluate(x),
            FormulaNode::Mul(a, b) => {
                let result = a.evaluate(x) * b.evaluate(x);
                if result.is_finite() {
                    result
                } else {
                    0.0
                }
            }
            FormulaNode::Div(a, b) => {
                let denom = b.evaluate(x);
                if denom.abs() < 1e-10 {
                    1.0
                } else {
                    let result = a.evaluate(x) / denom;
                    if result.is_finite() {
                        result
                    } else {
                        1.0
                    }
                }
            }
            FormulaNode::Sqrt(child) => {
                let val = child.evaluate(x).abs();
                val.sqrt()
            }
            FormulaNode::Log(child) => {
                let val = child.evaluate(x).abs() + 1.0;
                val.ln()
            }
            FormulaNode::Pow(base, exp) => {
                let b = base.evaluate(x);
                let e = exp.evaluate(x).clamp(-10.0, 10.0);
                let result = b.abs().powf(e);
                if result.is_finite() && result < 1e15 {
                    result
                } else {
                    0.0
                }
            }
        }
    }

    /// Count the number of nodes in this subtree.
    pub fn node_count(&self) -> usize {
        match self {
            FormulaNode::Const(_) | FormulaNode::Var => 1,
            FormulaNode::Add(a, b)
            | FormulaNode::Sub(a, b)
            | FormulaNode::Mul(a, b)
            | FormulaNode::Div(a, b)
            | FormulaNode::Pow(a, b) => 1 + a.node_count() + b.node_count(),
            FormulaNode::Sqrt(c) | FormulaNode::Log(c) => 1 + c.node_count(),
        }
    }
}

impl fmt::Display for FormulaNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FormulaNode::Const(c) => write!(f, "{:.2}", c),
            FormulaNode::Var => write!(f, "x"),
            FormulaNode::Add(a, b) => write!(f, "({} + {})", a, b),
            FormulaNode::Sub(a, b) => write!(f, "({} - {})", a, b),
            FormulaNode::Mul(a, b) => write!(f, "({} * {})", a, b),
            FormulaNode::Div(a, b) => write!(f, "({} / {})", a, b),
            FormulaNode::Sqrt(c) => write!(f, "sqrt({})", c),
            FormulaNode::Log(c) => write!(f, "ln({})", c),
            FormulaNode::Pow(base, exp) => write!(f, "({}^{})", base, exp),
        }
    }
}

// ---------------------------------------------------------------------------
// Data point for regression
// ---------------------------------------------------------------------------

/// A single observation for symbolic regression.
#[derive(Debug, Clone)]
pub struct DataPoint {
    /// Input (e.g., bit_size as f64).
    pub x: f64,
    /// Output (e.g., optimal parameter value, or success rate).
    pub y: f64,
}

// ---------------------------------------------------------------------------
// Formula generation and evolution
// ---------------------------------------------------------------------------

/// Generate a random formula tree with bounded depth.
pub fn random_formula(rng: &mut impl Rng, max_depth: u32) -> FormulaNode {
    if max_depth <= 1 {
        if rng.gen_bool(0.5) {
            FormulaNode::Var
        } else {
            // Constants from useful ranges
            let c = match rng.gen_range(0..5) {
                0 => rng.gen_range(0.1..10.0),
                1 => rng.gen_range(0.5..3.0),
                2 => 2.0 / 3.0, // Common exponent
                3 => 1.0 / 3.0,
                _ => rng.gen_range(-5.0..5.0),
            };
            FormulaNode::Const(c)
        }
    } else {
        match rng.gen_range(0..9) {
            0 => FormulaNode::Var,
            1 => FormulaNode::Const(rng.gen_range(0.1..10.0)),
            2 => FormulaNode::Add(
                Box::new(random_formula(rng, max_depth - 1)),
                Box::new(random_formula(rng, max_depth - 1)),
            ),
            3 => FormulaNode::Sub(
                Box::new(random_formula(rng, max_depth - 1)),
                Box::new(random_formula(rng, max_depth - 1)),
            ),
            4 => FormulaNode::Mul(
                Box::new(random_formula(rng, max_depth - 1)),
                Box::new(random_formula(rng, max_depth - 1)),
            ),
            5 => FormulaNode::Div(
                Box::new(random_formula(rng, max_depth - 1)),
                Box::new(random_formula(rng, max_depth - 1)),
            ),
            6 => FormulaNode::Sqrt(Box::new(random_formula(rng, max_depth - 1))),
            7 => FormulaNode::Log(Box::new(random_formula(rng, max_depth - 1))),
            _ => FormulaNode::Pow(
                Box::new(random_formula(rng, max_depth - 1)),
                Box::new(random_formula(rng, max_depth - 1)),
            ),
        }
    }
}

/// Mutate a formula by replacing a random subtree.
pub fn mutate_formula(formula: &FormulaNode, rng: &mut impl Rng) -> FormulaNode {
    // With some probability, replace the whole thing
    if rng.gen_bool(0.3) || formula.node_count() <= 1 {
        return random_formula(rng, 3);
    }

    match formula {
        FormulaNode::Const(c) => {
            // Tweak constant
            let delta = rng.gen_range(-2.0..2.0);
            FormulaNode::Const(c + delta)
        }
        FormulaNode::Var => random_formula(rng, 2),
        FormulaNode::Add(a, b) => {
            if rng.gen_bool(0.5) {
                FormulaNode::Add(Box::new(mutate_formula(a, rng)), b.clone())
            } else {
                FormulaNode::Add(a.clone(), Box::new(mutate_formula(b, rng)))
            }
        }
        FormulaNode::Sub(a, b) => {
            if rng.gen_bool(0.5) {
                FormulaNode::Sub(Box::new(mutate_formula(a, rng)), b.clone())
            } else {
                FormulaNode::Sub(a.clone(), Box::new(mutate_formula(b, rng)))
            }
        }
        FormulaNode::Mul(a, b) => {
            if rng.gen_bool(0.5) {
                FormulaNode::Mul(Box::new(mutate_formula(a, rng)), b.clone())
            } else {
                FormulaNode::Mul(a.clone(), Box::new(mutate_formula(b, rng)))
            }
        }
        FormulaNode::Div(a, b) => {
            if rng.gen_bool(0.5) {
                FormulaNode::Div(Box::new(mutate_formula(a, rng)), b.clone())
            } else {
                FormulaNode::Div(a.clone(), Box::new(mutate_formula(b, rng)))
            }
        }
        FormulaNode::Sqrt(c) => FormulaNode::Sqrt(Box::new(mutate_formula(c, rng))),
        FormulaNode::Log(c) => FormulaNode::Log(Box::new(mutate_formula(c, rng))),
        FormulaNode::Pow(base, exp) => {
            if rng.gen_bool(0.5) {
                FormulaNode::Pow(Box::new(mutate_formula(base, rng)), exp.clone())
            } else {
                FormulaNode::Pow(base.clone(), Box::new(mutate_formula(exp, rng)))
            }
        }
    }
}

/// Compute R² (coefficient of determination) for a formula on a dataset.
///
/// R² = 1 - SS_res / SS_tot, where:
/// - SS_res = Σ(y_i - f(x_i))²
/// - SS_tot = Σ(y_i - ȳ)²
///
/// Returns a value in (-∞, 1], where 1 is a perfect fit.
pub fn r_squared(formula: &FormulaNode, data: &[DataPoint]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }

    let mean_y: f64 = data.iter().map(|d| d.y).sum::<f64>() / data.len() as f64;

    let ss_tot: f64 = data.iter().map(|d| (d.y - mean_y).powi(2)).sum();
    if ss_tot < 1e-10 {
        return 0.0; // All y values are the same
    }

    let ss_res: f64 = data
        .iter()
        .map(|d| {
            let predicted = formula.evaluate(d.x);
            if predicted.is_finite() {
                (d.y - predicted).powi(2)
            } else {
                d.y.powi(2) // Penalize non-finite predictions
            }
        })
        .sum();

    1.0 - ss_res / ss_tot
}

/// A formula individual in the symbolic regression population.
struct FormulaIndividual {
    formula: FormulaNode,
    fitness: f64,
}

/// Run symbolic regression to find a formula fitting the data.
///
/// Uses tournament selection GP with the given population size and generations.
/// Returns the best formula found and its R² score.
pub fn symbolic_regression(
    data: &[DataPoint],
    pop_size: usize,
    generations: u32,
    rng: &mut impl Rng,
) -> (FormulaNode, f64) {
    if data.is_empty() {
        return (FormulaNode::Const(0.0), 0.0);
    }

    // Initialize population
    let mut population: Vec<FormulaIndividual> = (0..pop_size)
        .map(|_| {
            let formula = random_formula(rng, 4);
            let fitness = r_squared(&formula, data);
            FormulaIndividual { formula, fitness }
        })
        .collect();

    for _ in 0..generations {
        // Create offspring
        let num_offspring = pop_size / 4;
        let mut offspring: Vec<FormulaIndividual> = Vec::with_capacity(num_offspring);

        for _ in 0..num_offspring {
            // Tournament selection (size 3)
            let parent_idx = tournament_select_formula(&population, 3, rng);
            let mut child = mutate_formula(&population[parent_idx].formula, rng);

            // Bloat control: cap at 30 nodes
            if child.node_count() > 30 {
                child = random_formula(rng, 3);
            }

            let fitness = r_squared(&child, data);
            offspring.push(FormulaIndividual {
                formula: child,
                fitness,
            });
        }

        // Replace worst with offspring
        population.sort_by(|a, b| {
            a.fitness
                .partial_cmp(&b.fitness)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        for (i, child) in offspring.into_iter().enumerate() {
            if i < population.len() {
                population[i] = child;
            }
        }
    }

    // Return best
    population.sort_by(|a, b| {
        b.fitness
            .partial_cmp(&a.fitness)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let best = &population[0];
    (best.formula.clone(), best.fitness)
}

fn tournament_select_formula(
    population: &[FormulaIndividual],
    tournament_size: usize,
    rng: &mut impl Rng,
) -> usize {
    let mut best_idx = rng.gen_range(0..population.len());
    for _ in 1..tournament_size {
        let candidate = rng.gen_range(0..population.len());
        if population[candidate].fitness > population[best_idx].fitness {
            best_idx = candidate;
        }
    }
    best_idx
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_formula_evaluate_const() {
        let f = FormulaNode::Const(3.14);
        assert!((f.evaluate(0.0) - 3.14).abs() < 1e-10);
        assert!((f.evaluate(100.0) - 3.14).abs() < 1e-10);
    }

    #[test]
    fn test_formula_evaluate_var() {
        let f = FormulaNode::Var;
        assert!((f.evaluate(42.0) - 42.0).abs() < 1e-10);
    }

    #[test]
    fn test_formula_evaluate_add() {
        let f = FormulaNode::Add(
            Box::new(FormulaNode::Var),
            Box::new(FormulaNode::Const(10.0)),
        );
        assert!((f.evaluate(5.0) - 15.0).abs() < 1e-10);
    }

    #[test]
    fn test_formula_evaluate_mul() {
        let f = FormulaNode::Mul(
            Box::new(FormulaNode::Var),
            Box::new(FormulaNode::Const(3.0)),
        );
        assert!((f.evaluate(5.0) - 15.0).abs() < 1e-10);
    }

    #[test]
    fn test_formula_evaluate_protected_div() {
        let f = FormulaNode::Div(
            Box::new(FormulaNode::Const(10.0)),
            Box::new(FormulaNode::Const(0.0)),
        );
        assert!((f.evaluate(0.0) - 1.0).abs() < 1e-10); // Protected division
    }

    #[test]
    fn test_formula_display() {
        let f = FormulaNode::Mul(
            Box::new(FormulaNode::Const(3.0)),
            Box::new(FormulaNode::Pow(
                Box::new(FormulaNode::Var),
                Box::new(FormulaNode::Const(0.67)),
            )),
        );
        let s = format!("{}", f);
        assert!(s.contains("3.00"));
        assert!(s.contains("x"));
        assert!(s.contains("0.67"));
    }

    #[test]
    fn test_r_squared_perfect_fit() {
        // y = 2x
        let data: Vec<DataPoint> = (1..=10)
            .map(|x| DataPoint {
                x: x as f64,
                y: 2.0 * x as f64,
            })
            .collect();

        let formula = FormulaNode::Mul(
            Box::new(FormulaNode::Const(2.0)),
            Box::new(FormulaNode::Var),
        );

        let r2 = r_squared(&formula, &data);
        assert!(
            (r2 - 1.0).abs() < 1e-10,
            "Perfect fit should have R² = 1.0, got {}",
            r2
        );
    }

    #[test]
    fn test_r_squared_constant_prediction() {
        // y = x, predicted = constant (mean)
        let data: Vec<DataPoint> = (1..=10)
            .map(|x| DataPoint {
                x: x as f64,
                y: x as f64,
            })
            .collect();

        let mean = 5.5;
        let formula = FormulaNode::Const(mean);
        let r2 = r_squared(&formula, &data);
        assert!(
            r2.abs() < 1e-10,
            "Mean prediction should have R² ≈ 0, got {}",
            r2
        );
    }

    #[test]
    fn test_symbolic_regression_linear() {
        let mut rng = rand::thread_rng();

        // y = 3x + 2 with some noise
        let data: Vec<DataPoint> = (1..=20)
            .map(|x| DataPoint {
                x: x as f64,
                y: 3.0 * x as f64 + 2.0,
            })
            .collect();

        let (formula, r2) = symbolic_regression(&data, 200, 100, &mut rng);
        // Should find a decent fit for a simple linear relationship
        assert!(
            r2 > 0.5,
            "Should find R² > 0.5 for y=3x+2, got R²={:.4}, formula={}",
            r2,
            formula
        );
    }

    #[test]
    fn test_random_formula_valid() {
        let mut rng = rand::thread_rng();
        for _ in 0..20 {
            let formula = random_formula(&mut rng, 4);
            let result = formula.evaluate(10.0);
            assert!(
                result.is_finite() || result == f64::INFINITY || result == f64::NEG_INFINITY,
                "Formula should produce a numeric result"
            );
        }
    }

    #[test]
    fn test_node_count() {
        let f = FormulaNode::Add(
            Box::new(FormulaNode::Var),
            Box::new(FormulaNode::Const(1.0)),
        );
        assert_eq!(f.node_count(), 3);
    }
}
