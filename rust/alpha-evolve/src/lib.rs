//! Alpha-Evolve: Evolutionary algorithm search for novel factoring strategies.
//!
//! Uses genetic programming to discover factoring algorithms from composable
//! primitives. Programs are tree-structured combinations of modular arithmetic
//! operations that are evolved via mutation, crossover, and tournament selection.

pub mod evolution;
pub mod fitness;
pub mod primitives;

use num_bigint::BigUint;
use num_traits::{One, Zero};
use std::fmt;

use crate::primitives::{
    add_const_mod, fermat_step, gcd_prim, hart_step, is_perfect_square, isqrt, multiply_mod,
    random_element_prim, square_mod, subtract_gcd, williams_p_plus_1_step,
};

// ---------------------------------------------------------------------------
// DSL vocabulary
// ---------------------------------------------------------------------------

/// A primitive operation in the factoring DSL.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PrimitiveOp {
    /// Compute base^exp mod n.
    ModPow,
    /// Compute gcd(state, n); return if nontrivial.
    Gcd,
    /// Sample a random element in [2, n-2].
    RandomElement,
    /// Apply a sub-program repeatedly.
    Iterate { steps: u32 },
    /// Accumulate products, batch GCD.
    AccumulateGcd { batch_size: u32 },
    /// Compute gcd(|state - prev_state|, n).
    SubtractGcd,
    /// state = state^2 mod n.
    Square,
    /// state = (state + c) mod n.
    AddConst { c: u64 },
    /// state = (state * prev_state) mod n.
    MultiplyMod,
    /// Fermat step: compute a^2 - k*N, check perfect square.
    FermatStep { k: u64 },
    /// Hart's one-line: try multiplier i.
    HartStep,
    /// Williams p+1 step with Lucas sequence.
    WilliamsStep { bound: u64 },
    /// Integer square root of state.
    ISqrt,
    /// Fast perfect square test on state.
    IsPerfectSquare,
}

impl fmt::Display for PrimitiveOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PrimitiveOp::ModPow => write!(f, "ModPow"),
            PrimitiveOp::Gcd => write!(f, "Gcd"),
            PrimitiveOp::RandomElement => write!(f, "Rand"),
            PrimitiveOp::Iterate { steps } => write!(f, "Iter({})", steps),
            PrimitiveOp::AccumulateGcd { batch_size } => write!(f, "AccGcd({})", batch_size),
            PrimitiveOp::SubtractGcd => write!(f, "SubGcd"),
            PrimitiveOp::Square => write!(f, "Sq"),
            PrimitiveOp::AddConst { c } => write!(f, "Add({})", c),
            PrimitiveOp::MultiplyMod => write!(f, "Mul"),
            PrimitiveOp::FermatStep { k } => write!(f, "Fermat({})", k),
            PrimitiveOp::HartStep => write!(f, "Hart"),
            PrimitiveOp::WilliamsStep { bound } => write!(f, "Williams({})", bound),
            PrimitiveOp::ISqrt => write!(f, "ISqrt"),
            PrimitiveOp::IsPerfectSquare => write!(f, "IsSq"),
        }
    }
}

// ---------------------------------------------------------------------------
// Program tree
// ---------------------------------------------------------------------------

/// A node in the program tree.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProgramNode {
    /// A terminal operation.
    Leaf(PrimitiveOp),
    /// Execute children in order.
    Sequence(Vec<ProgramNode>),
    /// Loop body `steps` times.
    IterateNode {
        body: Box<ProgramNode>,
        steps: u32,
    },
    /// Run setup, then check gcd(state, n) for a nontrivial factor.
    GcdCheck {
        setup: Box<ProgramNode>,
    },
}

impl fmt::Display for ProgramNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ProgramNode::Leaf(op) => write!(f, "{}", op),
            ProgramNode::Sequence(children) => {
                write!(f, "Seq[")?;
                for (i, child) in children.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", child)?;
                }
                write!(f, "]")
            }
            ProgramNode::IterateNode { body, steps } => {
                write!(f, "Loop({}, {})", steps, body)
            }
            ProgramNode::GcdCheck { setup } => {
                write!(f, "GcdChk({})", setup)
            }
        }
    }
}

impl ProgramNode {
    /// Count the total number of nodes in this subtree.
    pub fn node_count(&self) -> usize {
        match self {
            ProgramNode::Leaf(_) => 1,
            ProgramNode::Sequence(children) => {
                1 + children.iter().map(|c| c.node_count()).collect::<Vec<_>>().iter().sum::<usize>()
            }
            ProgramNode::IterateNode { body, .. } => 1 + body.node_count(),
            ProgramNode::GcdCheck { setup } => 1 + setup.node_count(),
        }
    }

    /// Get a reference to the node at the given index (pre-order traversal).
    /// Returns None if index is out of range.
    pub fn get_node(&self, index: usize) -> Option<&ProgramNode> {
        let mut counter = 0;
        self.get_node_impl(index, &mut counter)
    }

    fn get_node_impl(&self, target: usize, counter: &mut usize) -> Option<&ProgramNode> {
        if *counter == target {
            return Some(self);
        }
        *counter += 1;

        match self {
            ProgramNode::Leaf(_) => None,
            ProgramNode::Sequence(children) => {
                for child in children {
                    if let Some(node) = child.get_node_impl(target, counter) {
                        return Some(node);
                    }
                }
                None
            }
            ProgramNode::IterateNode { body, .. } => body.get_node_impl(target, counter),
            ProgramNode::GcdCheck { setup } => setup.get_node_impl(target, counter),
        }
    }

    /// Replace the node at the given index with the replacement (pre-order traversal).
    /// Returns a new tree with the replacement inserted.
    pub fn replace_node(&self, index: usize, replacement: &ProgramNode) -> ProgramNode {
        let mut counter = 0;
        self.replace_node_impl(index, replacement, &mut counter)
    }

    fn replace_node_impl(
        &self,
        target: usize,
        replacement: &ProgramNode,
        counter: &mut usize,
    ) -> ProgramNode {
        if *counter == target {
            *counter += 1;
            // Skip counting the rest of the old subtree
            return replacement.clone();
        }
        *counter += 1;

        match self {
            ProgramNode::Leaf(op) => ProgramNode::Leaf(op.clone()),
            ProgramNode::Sequence(children) => {
                let new_children: Vec<ProgramNode> = children
                    .iter()
                    .map(|c| c.replace_node_impl(target, replacement, counter))
                    .collect();
                ProgramNode::Sequence(new_children)
            }
            ProgramNode::IterateNode { body, steps } => ProgramNode::IterateNode {
                body: Box::new(body.replace_node_impl(target, replacement, counter)),
                steps: *steps,
            },
            ProgramNode::GcdCheck { setup } => ProgramNode::GcdCheck {
                setup: Box::new(setup.replace_node_impl(target, replacement, counter)),
            },
        }
    }
}

// ---------------------------------------------------------------------------
// Program (executable)
// ---------------------------------------------------------------------------

/// An executable factoring program composed of primitive operations.
#[derive(Debug, Clone)]
pub struct Program {
    pub root: ProgramNode,
}

impl fmt::Display for Program {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.root)
    }
}

/// Internal execution state for the stack-based evaluator.
struct EvalState {
    /// Current working value (the "fast" pointer in Floyd-style iteration).
    state: BigUint,
    /// Previous working value (for MultiplyMod and general use).
    prev_state: BigUint,
    /// Slow pointer for cycle detection (updated every other iteration).
    slow_state: BigUint,
    /// Iteration counter within the current loop (for slow pointer updates).
    iter_count: u32,
    /// Accumulated product for batch GCD.
    accumulator: BigUint,
    /// Operation counter for timeout enforcement.
    ops: u32,
    /// Maximum operations before forced termination.
    max_ops: u32,
    /// Found factor (if any).
    found_factor: Option<BigUint>,
    /// The modulus (number to factor).
    n: BigUint,
    /// RNG for random operations.
    rng: rand::rngs::ThreadRng,
}

impl EvalState {
    fn new(n: &BigUint) -> Self {
        EvalState {
            state: BigUint::from(2u32),
            prev_state: BigUint::from(2u32),
            slow_state: BigUint::from(2u32),
            iter_count: 0,
            accumulator: BigUint::one(),
            ops: 0,
            max_ops: 1000,
            found_factor: None,
            n: n.clone(),
            rng: rand::thread_rng(),
        }
    }

    /// Increment the operation counter. Returns true if we should stop.
    fn tick(&mut self) -> bool {
        self.ops += 1;
        self.ops >= self.max_ops || self.found_factor.is_some()
    }

    /// Check if gcd(state, n) is a nontrivial factor.
    fn check_gcd(&mut self) {
        let g = gcd_prim(&self.state, &self.n);
        let one = BigUint::one();
        if g > one && g < self.n {
            self.found_factor = Some(g);
        }
    }
}

impl Program {
    /// Execute the program tree on the target number `n`.
    /// Returns a nontrivial factor if one is discovered, or None.
    /// Enforces a hard limit of 1000 operations to prevent infinite loops.
    pub fn evaluate(&self, n: &BigUint) -> Option<BigUint> {
        let one = BigUint::one();
        if *n <= one {
            return None;
        }

        let mut eval = EvalState::new(n);
        execute_node(&self.root, &mut eval);
        eval.found_factor
    }
}

/// Recursively execute a program node, updating the evaluation state.
/// Only leaf operations consume the operation budget; structural nodes
/// (Sequence, IterateNode, GcdCheck) are free to allow deeper programs.
fn execute_node(node: &ProgramNode, eval: &mut EvalState) {
    if eval.found_factor.is_some() || eval.ops >= eval.max_ops {
        return;
    }

    match node {
        ProgramNode::Leaf(op) => {
            if eval.tick() {
                return;
            }
            execute_op(op, eval);
        }
        ProgramNode::Sequence(children) => {
            for child in children {
                execute_node(child, eval);
                if eval.found_factor.is_some() || eval.ops >= eval.max_ops {
                    return;
                }
            }
        }
        ProgramNode::IterateNode { body, steps } => {
            // For cycle detection (SubtractGcd), maintain a slow pointer.
            // The slow pointer advances by applying the same body every other
            // iteration (Floyd's tortoise-and-hare).
            eval.slow_state = eval.state.clone();
            eval.iter_count = 0;

            for _ in 0..*steps {
                eval.prev_state = eval.state.clone();

                // Advance the fast pointer (main state) by executing the body
                execute_node(body, eval);
                eval.iter_count += 1;

                // Advance the slow pointer every other iteration by
                // re-executing the body with slow_state as the state.
                if eval.iter_count % 2 == 0 {
                    let saved_state = eval.state.clone();
                    let saved_prev = eval.prev_state.clone();
                    eval.state = eval.slow_state.clone();
                    execute_node(body, eval);
                    eval.slow_state = eval.state.clone();
                    eval.state = saved_state;
                    eval.prev_state = saved_prev;
                }

                if eval.found_factor.is_some() || eval.ops >= eval.max_ops {
                    return;
                }
            }
        }
        ProgramNode::GcdCheck { setup } => {
            execute_node(setup, eval);
            if eval.found_factor.is_none() && eval.ops < eval.max_ops {
                eval.check_gcd();
            }
        }
    }
}

/// Execute a single primitive operation.
fn execute_op(op: &PrimitiveOp, eval: &mut EvalState) {
    match op {
        PrimitiveOp::ModPow => {
            // Compute state^prev_state mod n (a generic modpow with available values)
            let exp = if eval.prev_state.is_zero() {
                BigUint::from(2u32)
            } else {
                // Clamp exponent to avoid enormous computation
                let two = BigUint::from(2u32);
                if eval.prev_state.bits() > 20 {
                    two
                } else {
                    eval.prev_state.clone()
                }
            };
            eval.prev_state = eval.state.clone();
            eval.state = eval.state.modpow(&exp, &eval.n);
        }
        PrimitiveOp::Gcd => {
            eval.check_gcd();
        }
        PrimitiveOp::RandomElement => {
            eval.prev_state = eval.state.clone();
            eval.state = random_element_prim(&eval.n, &mut eval.rng);
            // Also reset slow_state so cycle detection starts fresh
            eval.slow_state = eval.state.clone();
        }
        PrimitiveOp::Iterate { steps } => {
            // As a leaf, iterate just means "repeat the last operation" conceptually.
            // In practice, we square `steps` times (a common sub-routine).
            for _ in 0..*steps {
                if eval.ops >= eval.max_ops || eval.found_factor.is_some() {
                    return;
                }
                eval.ops += 1;
                eval.prev_state = eval.state.clone();
                eval.state = square_mod(&eval.state, &eval.n);
            }
        }
        PrimitiveOp::AccumulateGcd { batch_size } => {
            // Accumulate state into the product, check gcd every batch_size accumulations
            eval.accumulator = multiply_mod(&eval.accumulator, &eval.state, &eval.n);
            if *batch_size > 0 && eval.ops % batch_size == 0 {
                let g = gcd_prim(&eval.accumulator, &eval.n);
                let one = BigUint::one();
                if g > one && g < eval.n {
                    eval.found_factor = Some(g);
                }
                eval.accumulator = BigUint::one();
            }
        }
        PrimitiveOp::SubtractGcd => {
            // Compare the current (fast) state with the slow state for
            // Floyd-style cycle detection: gcd(|fast - slow|, n).
            if let Some(factor) = subtract_gcd(&eval.state, &eval.slow_state, &eval.n) {
                eval.found_factor = Some(factor);
            }
        }
        PrimitiveOp::Square => {
            eval.prev_state = eval.state.clone();
            eval.state = square_mod(&eval.state, &eval.n);
        }
        PrimitiveOp::AddConst { c } => {
            eval.prev_state = eval.state.clone();
            eval.state = add_const_mod(&eval.state, *c, &eval.n);
        }
        PrimitiveOp::MultiplyMod => {
            let result = multiply_mod(&eval.state, &eval.prev_state, &eval.n);
            eval.prev_state = eval.state.clone();
            eval.state = result;
        }
        PrimitiveOp::FermatStep { k } => {
            // Use state as 'a' value for Fermat
            if let Some(factor) = fermat_step(&eval.n, &eval.state, *k) {
                eval.found_factor = Some(factor);
            }
            // Increment state for next attempt
            eval.state = add_const_mod(&eval.state, 1, &eval.n);
        }
        PrimitiveOp::HartStep => {
            // Use iter_count as multiplier
            let i = (eval.iter_count as u64).max(1);
            if let Some(factor) = hart_step(&eval.n, i) {
                eval.found_factor = Some(factor);
            }
        }
        PrimitiveOp::WilliamsStep { bound } => {
            if let Some(factor) = williams_p_plus_1_step(&eval.n, &eval.state, *bound) {
                eval.found_factor = Some(factor);
            }
        }
        PrimitiveOp::ISqrt => {
            eval.prev_state = eval.state.clone();
            eval.state = isqrt(&eval.state);
        }
        PrimitiveOp::IsPerfectSquare => {
            // Set state to 1 if perfect square, 0 otherwise (useful as a condition)
            eval.prev_state = eval.state.clone();
            if is_perfect_square(&eval.state) {
                eval.state = BigUint::one();
            } else {
                eval.state = BigUint::zero();
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Individual (for the population)
// ---------------------------------------------------------------------------

/// An individual in the evolutionary population: a program paired with its fitness score.
#[derive(Debug, Clone)]
pub struct Individual {
    pub program: Program,
    pub fitness: f64,
}

// ---------------------------------------------------------------------------
// Seed programs (hand-built known algorithms)
// ---------------------------------------------------------------------------

/// Build a Pollard's rho-like program:
/// random start, iterate x = x^2 + c, gcd check every step.
pub fn seed_pollard_rho() -> Program {
    // Structure: Seq[ Rand, Loop(200, Seq[Sq, Add(1), GcdChk(Gcd)]) ]
    let inner = ProgramNode::Sequence(vec![
        ProgramNode::Leaf(PrimitiveOp::Square),
        ProgramNode::Leaf(PrimitiveOp::AddConst { c: 1 }),
        ProgramNode::Leaf(PrimitiveOp::SubtractGcd),
    ]);

    let root = ProgramNode::Sequence(vec![
        ProgramNode::Leaf(PrimitiveOp::RandomElement),
        ProgramNode::IterateNode {
            body: Box::new(inner),
            steps: 200,
        },
    ]);

    Program { root }
}

/// Build a trial-division-like program:
/// iterate through small values, gcd check each.
pub fn seed_trial_like() -> Program {
    // Structure: Seq[ Add(2), Loop(100, Seq[GcdChk(Gcd), Add(1)]) ]
    // Start at 2, increment by 1 each iteration, check gcd each time.
    let inner = ProgramNode::Sequence(vec![
        ProgramNode::GcdCheck {
            setup: Box::new(ProgramNode::Leaf(PrimitiveOp::Gcd)),
        },
        ProgramNode::Leaf(PrimitiveOp::AddConst { c: 1 }),
    ]);

    let root = ProgramNode::Sequence(vec![
        ProgramNode::Leaf(PrimitiveOp::AddConst { c: 0 }), // state starts at 2, add 0 = still 2
        ProgramNode::IterateNode {
            body: Box::new(inner),
            steps: 200,
        },
    ]);

    Program { root }
}

/// Build a Fermat-like program:
/// start near sqrt(n), square and subtract, gcd check.
/// Since we cannot compute sqrt easily in the DSL, we use ModPow to
/// approximate: state = random, then iterate (state^2 - n) and check gcd.
/// In practice we just do: random start, iterate Square + SubtractGcd.
pub fn seed_fermat_like() -> Program {
    // Structure: Seq[ Rand, Loop(200, Seq[Sq, SubGcd]) ]
    let inner = ProgramNode::Sequence(vec![
        ProgramNode::Leaf(PrimitiveOp::Square),
        ProgramNode::Leaf(PrimitiveOp::SubtractGcd),
    ]);

    let root = ProgramNode::Sequence(vec![
        ProgramNode::Leaf(PrimitiveOp::RandomElement),
        ProgramNode::IterateNode {
            body: Box::new(inner),
            steps: 200,
        },
    ]);

    Program { root }
}

/// Build a Lehman-like program: sqrt(n), then Fermat steps with multipliers.
pub fn seed_lehman_like() -> Program {
    let inner = ProgramNode::Sequence(vec![
        ProgramNode::Leaf(PrimitiveOp::FermatStep { k: 1 }),
        ProgramNode::Leaf(PrimitiveOp::AddConst { c: 1 }),
    ]);

    let root = ProgramNode::Sequence(vec![
        ProgramNode::Leaf(PrimitiveOp::ISqrt), // start near sqrt(n)
        ProgramNode::IterateNode {
            body: Box::new(inner),
            steps: 200,
        },
    ]);

    Program { root }
}

/// Build a Hart-like program: iterate through multipliers.
pub fn seed_hart_like() -> Program {
    let inner = ProgramNode::Sequence(vec![
        ProgramNode::Leaf(PrimitiveOp::HartStep),
        ProgramNode::Leaf(PrimitiveOp::AddConst { c: 1 }),
    ]);

    let root = ProgramNode::Sequence(vec![
        ProgramNode::Leaf(PrimitiveOp::AddConst { c: 0 }), // state starts at 2
        ProgramNode::IterateNode {
            body: Box::new(inner),
            steps: 200,
        },
    ]);

    Program { root }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_seed_rho_factors() {
        // 8051 = 83 * 97
        let n = BigUint::from(8051u32);
        // Pollard rho is probabilistic; try multiple times
        let program = seed_pollard_rho();
        let mut found = false;
        for _ in 0..20 {
            if let Some(factor) = program.evaluate(&n) {
                assert!(
                    factor == BigUint::from(83u32) || factor == BigUint::from(97u32),
                    "Expected factor 83 or 97, got {}",
                    factor
                );
                assert!((&n % &factor).is_zero());
                found = true;
                break;
            }
        }
        assert!(found, "seed_pollard_rho should factor 8051 within 20 attempts");
    }

    #[test]
    fn test_program_display() {
        let prog = seed_pollard_rho();
        let desc = format!("{}", prog);
        assert!(!desc.is_empty());
    }

    #[test]
    fn test_program_node_count() {
        let prog = seed_pollard_rho();
        assert!(prog.root.node_count() > 1);
    }

    #[test]
    fn test_program_evaluate_trivial() {
        // Evaluating on n=1 should return None
        let prog = seed_pollard_rho();
        assert!(prog.evaluate(&BigUint::one()).is_none());
    }

    #[test]
    fn test_trial_like_seed_factors_small() {
        // 15 = 3 * 5 -- trial-like should find this
        let n = BigUint::from(15u32);
        let program = seed_trial_like();
        let mut found = false;
        for _ in 0..5 {
            if let Some(factor) = program.evaluate(&n) {
                assert!(
                    factor == BigUint::from(3u32) || factor == BigUint::from(5u32),
                    "Expected factor 3 or 5, got {}",
                    factor
                );
                found = true;
                break;
            }
        }
        assert!(found, "seed_trial_like should factor 15");
    }
}
