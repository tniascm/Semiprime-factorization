//! Alpha-Evolve: Evolutionary algorithm search for novel factoring strategies.
//!
//! Uses genetic programming to discover factoring algorithms from composable
//! primitives. Programs are tree-structured combinations of modular arithmetic
//! operations that are evolved via mutation, crossover, and tournament selection.

pub mod evolution;
pub mod fitness;
pub mod macros;
pub mod novelty;
pub mod primitives;
pub mod symreg;

use num_bigint::BigUint;
use num_traits::{One, Zero};
use std::fmt;

use crate::primitives::{
    add_const_mod, cf_convergent_gcd, dixon_accumulate, dixon_combine, ecm_attempt,
    fermat_step, gcd_prim, hart_step, is_perfect_square, isqrt, jacobi_symbol_prim,
    lll_short_vector_gcd, multiply_mod, pilatte_vector_product, pollard_pm1_step,
    random_element_prim, rho_form_step, smooth_test, square_mod, squfof_attempt, subtract_gcd,
    williams_p_plus_1_step, DixonState,
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

    // --- New primitives from project crates ---

    /// k-th CF convergent of sqrt(N): compute gcd(p_k² - N, N).
    CfConvergent { k: u32 },
    /// Full SQUFOF factoring attempt with step budget.
    SqufofStep,
    /// One rho-step in the class group infrastructure of Q(sqrt(N)).
    RhoFormStep,
    /// Run one ECM curve with Phase 1 bound B1.
    EcmCurve { b1: u64 },
    /// Build a small lattice from [state, N], LLL-reduce, return gcd of short vector with N.
    LllShortVector,
    /// Test if state is B-smooth; if yes, return largest smooth factor.
    SmoothTest { bound: u64 },
    /// Generate one exponent vector from Pilatte lattice, compute product mod N.
    PilatteVector,
    /// Compute Jacobi symbol (state / N): sets state to 0, 1, or N-1.
    QuadraticResidue,
    /// Pollard p-1 step with smoothness bound B.
    PollardPm1 { bound: u64 },
    /// Accumulate a smooth square for Dixon's method.
    DixonAccumulate,
    /// Combine accumulated Dixon squares, try gcd.
    DixonCombine,
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
            PrimitiveOp::CfConvergent { k } => write!(f, "CF({})", k),
            PrimitiveOp::SqufofStep => write!(f, "SQUFOF"),
            PrimitiveOp::RhoFormStep => write!(f, "RhoForm"),
            PrimitiveOp::EcmCurve { b1 } => write!(f, "ECM({})", b1),
            PrimitiveOp::LllShortVector => write!(f, "LLL"),
            PrimitiveOp::SmoothTest { bound } => write!(f, "Smooth({})", bound),
            PrimitiveOp::PilatteVector => write!(f, "Pilatte"),
            PrimitiveOp::QuadraticResidue => write!(f, "Jacobi"),
            PrimitiveOp::PollardPm1 { bound } => write!(f, "Pm1({})", bound),
            PrimitiveOp::DixonAccumulate => write!(f, "DixAcc"),
            PrimitiveOp::DixonCombine => write!(f, "DixCmb"),
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
    /// Conditional: if state > threshold, execute if_true; else execute if_false.
    ConditionalGt {
        threshold: u64,
        if_true: Box<ProgramNode>,
        if_false: Box<ProgramNode>,
    },
    /// Store current state into memory slot (0-3), or load from slot into state.
    MemoryOp {
        store: bool,
        slot: u8,
    },
    /// A macro algorithm block: runs a complete algorithm with evolved parameters.
    /// State is passed as a hint to the macro. If the macro finds a factor, it's
    /// set as the result. Otherwise, state is left unchanged.
    MacroBlock {
        kind: crate::macros::MacroKind,
        params: crate::macros::MacroParams,
    },
    /// Hybrid composition: run child A first; if it doesn't find a factor,
    /// feed the resulting state into child B.
    Hybrid {
        first: Box<ProgramNode>,
        second: Box<ProgramNode>,
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
            ProgramNode::ConditionalGt {
                threshold,
                if_true,
                if_false,
            } => {
                write!(f, "If(>{}, {}, {})", threshold, if_true, if_false)
            }
            ProgramNode::MemoryOp { store, slot } => {
                if *store {
                    write!(f, "Store({})", slot)
                } else {
                    write!(f, "Load({})", slot)
                }
            }
            ProgramNode::MacroBlock { kind, params } => {
                write!(f, "{}({},{})", kind, params.param1, params.param2)
            }
            ProgramNode::Hybrid { first, second } => {
                write!(f, "Hybrid({}, {})", first, second)
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
            ProgramNode::ConditionalGt {
                if_true, if_false, ..
            } => 1 + if_true.node_count() + if_false.node_count(),
            ProgramNode::MemoryOp { .. } => 1,
            ProgramNode::MacroBlock { .. } => 1,
            ProgramNode::Hybrid { first, second } => {
                1 + first.node_count() + second.node_count()
            }
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
            ProgramNode::Leaf(_) | ProgramNode::MemoryOp { .. } | ProgramNode::MacroBlock { .. } => None,
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
            ProgramNode::ConditionalGt {
                if_true, if_false, ..
            } => {
                if let Some(node) = if_true.get_node_impl(target, counter) {
                    return Some(node);
                }
                if_false.get_node_impl(target, counter)
            }
            ProgramNode::Hybrid { first, second } => {
                if let Some(node) = first.get_node_impl(target, counter) {
                    return Some(node);
                }
                second.get_node_impl(target, counter)
            }
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
            ProgramNode::MemoryOp { store, slot } => ProgramNode::MemoryOp {
                store: *store,
                slot: *slot,
            },
            ProgramNode::MacroBlock { kind, params } => ProgramNode::MacroBlock {
                kind: kind.clone(),
                params: params.clone(),
            },
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
            ProgramNode::ConditionalGt {
                threshold,
                if_true,
                if_false,
            } => ProgramNode::ConditionalGt {
                threshold: *threshold,
                if_true: Box::new(if_true.replace_node_impl(target, replacement, counter)),
                if_false: Box::new(if_false.replace_node_impl(target, replacement, counter)),
            },
            ProgramNode::Hybrid { first, second } => ProgramNode::Hybrid {
                first: Box::new(first.replace_node_impl(target, replacement, counter)),
                second: Box::new(second.replace_node_impl(target, replacement, counter)),
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
    /// Memory slots for storing/recalling intermediate values.
    memory: [BigUint; 4],
    /// Dixon method accumulation state.
    dixon: DixonState,
    /// Wall-clock deadline for the entire evaluation.
    deadline: std::time::Instant,
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
            memory: [
                BigUint::zero(),
                BigUint::zero(),
                BigUint::zero(),
                BigUint::zero(),
            ],
            dixon: DixonState::new(),
            deadline: std::time::Instant::now() + std::time::Duration::from_millis(50),
        }
    }

    /// Increment the operation counter. Returns true if we should stop.
    fn tick(&mut self) -> bool {
        self.ops += 1;
        self.ops >= self.max_ops
            || self.found_factor.is_some()
            || std::time::Instant::now() >= self.deadline
    }

    /// Check if the evaluation has exceeded its wall-clock deadline.
    fn timed_out(&self) -> bool {
        std::time::Instant::now() >= self.deadline
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
    if eval.found_factor.is_some() || eval.ops >= eval.max_ops || eval.timed_out() {
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
                if eval.found_factor.is_some() || eval.ops >= eval.max_ops || eval.timed_out() {
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

                if eval.found_factor.is_some() || eval.ops >= eval.max_ops || eval.timed_out() {
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
        ProgramNode::ConditionalGt {
            threshold,
            if_true,
            if_false,
        } => {
            let threshold_big = BigUint::from(*threshold);
            if eval.state > threshold_big {
                execute_node(if_true, eval);
            } else {
                execute_node(if_false, eval);
            }
        }
        ProgramNode::MemoryOp { store, slot } => {
            let idx = (*slot as usize) % 4;
            if *store {
                eval.memory[idx] = eval.state.clone();
            } else {
                eval.prev_state = eval.state.clone();
                eval.state = eval.memory[idx].clone();
            }
        }
        ProgramNode::MacroBlock { kind, params } => {
            if eval.tick() {
                return;
            }
            if let Some(factor) = crate::macros::execute_macro(
                kind,
                params,
                &eval.n,
                &eval.state,
            ) {
                let one = BigUint::one();
                if factor > one && factor < eval.n {
                    eval.found_factor = Some(factor);
                }
            }
        }
        ProgramNode::Hybrid { first, second } => {
            // Run the first child
            execute_node(first, eval);
            // If no factor found, run the second child with the state from the first
            if eval.found_factor.is_none() && !eval.timed_out() && eval.ops < eval.max_ops {
                execute_node(second, eval);
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
                if eval.ops >= eval.max_ops || eval.found_factor.is_some() || eval.timed_out() {
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

        // --- New primitives from project crates ---

        PrimitiveOp::CfConvergent { k } => {
            if let Some(factor) = cf_convergent_gcd(&eval.n, *k) {
                eval.found_factor = Some(factor);
            }
        }
        PrimitiveOp::SqufofStep => {
            if let Some(factor) = squfof_attempt(&eval.n) {
                eval.found_factor = Some(factor);
            }
        }
        PrimitiveOp::RhoFormStep => {
            if let Some(factor) = rho_form_step(&eval.n) {
                eval.found_factor = Some(factor);
            } else {
                // Update state with a value derived from the infrastructure
                eval.prev_state = eval.state.clone();
                eval.state = add_const_mod(&eval.state, 1, &eval.n);
            }
        }
        PrimitiveOp::EcmCurve { b1 } => {
            if let Some(factor) = ecm_attempt(&eval.n, *b1) {
                eval.found_factor = Some(factor);
            }
        }
        PrimitiveOp::LllShortVector => {
            if let Some(factor) = lll_short_vector_gcd(&eval.state, &eval.n) {
                eval.found_factor = Some(factor);
            }
        }
        PrimitiveOp::SmoothTest { bound } => {
            let result = smooth_test(&eval.state, *bound);
            eval.prev_state = eval.state.clone();
            eval.state = result;
        }
        PrimitiveOp::PilatteVector => {
            let product = pilatte_vector_product(&eval.n);
            eval.prev_state = eval.state.clone();
            eval.state = product;
            // Also try gcd
            let g = gcd_prim(&eval.state, &eval.n);
            let one = BigUint::one();
            if g > one && g < eval.n {
                eval.found_factor = Some(g);
            }
        }
        PrimitiveOp::QuadraticResidue => {
            eval.prev_state = eval.state.clone();
            eval.state = jacobi_symbol_prim(&eval.state, &eval.n);
        }
        PrimitiveOp::PollardPm1 { bound } => {
            if let Some(factor) = pollard_pm1_step(&eval.n, &eval.state, *bound) {
                eval.found_factor = Some(factor);
            }
            // Advance state
            eval.prev_state = eval.state.clone();
            eval.state = add_const_mod(&eval.state, 1, &eval.n);
        }
        PrimitiveOp::DixonAccumulate => {
            dixon_accumulate(&mut eval.dixon, &eval.state, &eval.n);
        }
        PrimitiveOp::DixonCombine => {
            if let Some(factor) = dixon_combine(&mut eval.dixon, &eval.n) {
                eval.found_factor = Some(factor);
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

// ---------------------------------------------------------------------------
// New seed programs from E1-E20 domain knowledge
// ---------------------------------------------------------------------------

/// Seed: Dixon-style smooth square accumulation.
///
/// Inspired by E13 (Eisenstein congruences): accumulate smooth squares,
/// then combine them to find congruences of squares.
/// Structure: Seq[ Rand, Loop(200, Seq[DixonAcc, DixonCmb]) ]
pub fn seed_dixon_smooth() -> Program {
    let inner = ProgramNode::Sequence(vec![
        ProgramNode::Leaf(PrimitiveOp::Square),
        ProgramNode::Leaf(PrimitiveOp::DixonAccumulate),
        ProgramNode::Leaf(PrimitiveOp::AddConst { c: 1 }),
        ProgramNode::Leaf(PrimitiveOp::DixonCombine),
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

/// Seed: CF convergent + regulator-guided jump.
///
/// Inspired by Prong 4 (Murru-Salvatori regulator-guided factoring):
/// use CF convergents to estimate the regulator, then walk the class
/// group infrastructure to find ambiguous forms.
/// Structure: Seq[ CF(10), Store(0), CF(20), Store(1), MacroClassWalk(500) ]
pub fn seed_cf_regulator_jump() -> Program {
    let root = ProgramNode::Sequence(vec![
        // Compute early CF convergent
        ProgramNode::Leaf(PrimitiveOp::CfConvergent { k: 10 }),
        ProgramNode::MemoryOp {
            store: true,
            slot: 0,
        },
        // Compute later CF convergent
        ProgramNode::Leaf(PrimitiveOp::CfConvergent { k: 30 }),
        ProgramNode::MemoryOp {
            store: true,
            slot: 1,
        },
        // Walk class group infrastructure
        ProgramNode::MacroBlock {
            kind: macros::MacroKind::ClassWalk,
            params: macros::MacroParams {
                param1: 500,
                param2: 1,
            },
        },
        // Fall back to SQUFOF
        ProgramNode::Leaf(PrimitiveOp::SqufofStep),
    ]);

    Program { root }
}

/// Seed: Lattice-based factoring via Pilatte short vectors.
///
/// Inspired by Prong 2 (smooth-pilatte): build a Pilatte lattice,
/// extract short vectors, compute products, and check gcd.
/// Structure: Seq[ PilatteVec, GcdChk, MacroLattice(6) ]
pub fn seed_lattice_gcd() -> Program {
    let root = ProgramNode::Sequence(vec![
        ProgramNode::Leaf(PrimitiveOp::PilatteVector),
        ProgramNode::GcdCheck {
            setup: Box::new(ProgramNode::Leaf(PrimitiveOp::LllShortVector)),
        },
        ProgramNode::MacroBlock {
            kind: macros::MacroKind::LatticeSmooth,
            params: macros::MacroParams {
                param1: 6,
                param2: 1,
            },
        },
    ]);

    Program { root }
}

/// Seed: ECM + CF hybrid.
///
/// Novel composition: use CF convergent as ECM curve seed, then run ECM.
/// The idea is that CF convergents of sqrt(N) produce algebraic values
/// structurally tied to N that might make better curve parameters.
/// Structure: Seq[ CF(5), Store(0), ECM(500), Load(0), ECM(1000) ]
pub fn seed_ecm_cf_hybrid() -> Program {
    let root = ProgramNode::Sequence(vec![
        ProgramNode::Leaf(PrimitiveOp::CfConvergent { k: 5 }),
        ProgramNode::MemoryOp {
            store: true,
            slot: 0,
        },
        ProgramNode::Leaf(PrimitiveOp::EcmCurve { b1: 500 }),
        ProgramNode::MemoryOp {
            store: false,
            slot: 0,
        },
        ProgramNode::Leaf(PrimitiveOp::EcmCurve { b1: 1000 }),
    ]);

    Program { root }
}

/// Seed: Pollard p-1 + rho hybrid via Hybrid node.
///
/// Inspired by BSGS infrastructure: try Pollard p-1 first (works when
/// p-1 is smooth), then fall back to Pollard rho.
/// Structure: Hybrid[ Pm1(200), Seq[Rand, Loop(100, Seq[Sq, Add(1), SubGcd])] ]
pub fn seed_pm1_rho_hybrid() -> Program {
    let pm1_branch = ProgramNode::Leaf(PrimitiveOp::PollardPm1 { bound: 200 });

    let rho_inner = ProgramNode::Sequence(vec![
        ProgramNode::Leaf(PrimitiveOp::Square),
        ProgramNode::Leaf(PrimitiveOp::AddConst { c: 1 }),
        ProgramNode::Leaf(PrimitiveOp::SubtractGcd),
    ]);

    let rho_branch = ProgramNode::Sequence(vec![
        ProgramNode::Leaf(PrimitiveOp::RandomElement),
        ProgramNode::IterateNode {
            body: Box::new(rho_inner),
            steps: 100,
        },
    ]);

    let root = ProgramNode::Hybrid {
        first: Box::new(pm1_branch),
        second: Box::new(rho_branch),
    };

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
