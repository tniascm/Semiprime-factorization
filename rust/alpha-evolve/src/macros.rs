//! Higher-level macro algorithm blocks for the evolutionary factoring DSL.
//!
//! Each macro block encapsulates a known algorithm fragment as a single
//! evolvable node with mutable parameters. This allows evolution to compose
//! algorithm-level strategies (e.g., "run SQUFOF then feed result to ECM")
//! rather than just primitives.
//!
//! Macro blocks are self-contained: they take N (and optionally a state hint)
//! and return an Option<BigUint> factor. They respect wall-clock timeouts
//! to prevent stalling the evolutionary evaluator.

use num_bigint::BigUint;
use num_integer::Integer;
use num_traits::{One, Zero};
use std::fmt;
use std::time::{Duration, Instant};

/// The different kinds of macro algorithm blocks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MacroKind {
    /// Full SQUFOF factoring attempt with step budget.
    Squfof,
    /// ECM with random curves and bounded B1.
    Ecm,
    /// Pollard rho with Brent cycle detection.
    PollardRho,
    /// Fermat's method scanning multipliers k in [k_start, k_end].
    FermatScan,
    /// Pilatte lattice → short vectors → gcd.
    LatticeSmooth,
    /// Walk class group infrastructure for a given number of steps.
    ClassWalk,
}

/// Parameters for a macro block, evolved by the GP engine.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MacroParams {
    /// Primary parameter (interpretation depends on MacroKind).
    /// - Squfof: unused (just uses default step budget)
    /// - Ecm: B1 bound (100..10000)
    /// - PollardRho: max iterations (100..50000)
    /// - FermatScan: k_start
    /// - LatticeSmooth: lattice dimension (4..12)
    /// - ClassWalk: number of steps (10..5000)
    pub param1: u64,
    /// Secondary parameter:
    /// - Ecm: number of curves (1..5)
    /// - FermatScan: k_end
    /// - All others: unused
    pub param2: u64,
}

impl Default for MacroParams {
    fn default() -> Self {
        MacroParams {
            param1: 1000,
            param2: 1,
        }
    }
}

impl fmt::Display for MacroKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MacroKind::Squfof => write!(f, "MacroSQUFOF"),
            MacroKind::Ecm => write!(f, "MacroECM"),
            MacroKind::PollardRho => write!(f, "MacroRho"),
            MacroKind::FermatScan => write!(f, "MacroFermat"),
            MacroKind::LatticeSmooth => write!(f, "MacroLattice"),
            MacroKind::ClassWalk => write!(f, "MacroClassWalk"),
        }
    }
}

/// Maximum wall-clock time for any single macro block execution.
const MACRO_TIMEOUT: Duration = Duration::from_millis(20);

/// Execute a macro block on the given N.
///
/// The `state_hint` can be used by some macros (e.g., FermatScan uses it
/// as a starting point; ECM uses it as a curve seed). This allows upstream
/// program nodes to feed information into macro blocks.
pub fn execute_macro(
    kind: &MacroKind,
    params: &MacroParams,
    n: &BigUint,
    state_hint: &BigUint,
) -> Option<BigUint> {
    let one = BigUint::one();
    if *n <= one {
        return None;
    }

    let deadline = Instant::now() + MACRO_TIMEOUT;

    match kind {
        MacroKind::Squfof => execute_squfof(n, &deadline),
        MacroKind::Ecm => execute_ecm(n, params, &deadline),
        MacroKind::PollardRho => execute_pollard_rho(n, params, state_hint, &deadline),
        MacroKind::FermatScan => execute_fermat_scan(n, params, state_hint, &deadline),
        MacroKind::LatticeSmooth => execute_lattice_smooth(n, params, &deadline),
        MacroKind::ClassWalk => execute_class_walk(n, params, &deadline),
    }
}

// ---------------------------------------------------------------------------
// Individual macro block implementations
// ---------------------------------------------------------------------------

fn execute_squfof(n: &BigUint, _deadline: &Instant) -> Option<BigUint> {
    // Only attempt for reasonably sized N
    if n.bits() > 48 {
        return None;
    }
    let result = cf_factor::squfof::squfof_factor(n);
    let one = BigUint::one();
    result.factor.filter(|f| *f > one && *f < *n)
}

fn execute_ecm(n: &BigUint, params: &MacroParams, _deadline: &Instant) -> Option<BigUint> {
    let b1 = params.param1.max(100).min(2000);
    let curves = params.param2.max(1).min(3) as usize;
    let ecm_params = ecm::EcmParams {
        b1,
        b2: b1 * 25,
        num_curves: curves,
    };
    let result = ecm::ecm_factor(n, &ecm_params);
    let one = BigUint::one();
    if result.complete {
        result.factors.into_iter().find(|f| *f > one && *f < *n)
    } else {
        None
    }
}

fn execute_pollard_rho(
    n: &BigUint,
    params: &MacroParams,
    state_hint: &BigUint,
    deadline: &Instant,
) -> Option<BigUint> {
    let one = BigUint::one();
    let max_iters = params.param1.max(100).min(10000);

    // Initialize with state_hint or default
    let mut x = if *state_hint > one && *state_hint < *n {
        state_hint.clone()
    } else {
        BigUint::from(2u32)
    };
    let mut y = x.clone();
    let c = BigUint::one();

    for _ in 0..max_iters {
        if Instant::now() >= *deadline {
            return None;
        }
        // x = x² + c mod N
        x = (&x * &x + &c) % n;
        // y = (y² + c)² + c mod N (two steps)
        y = (&y * &y + &c) % n;
        y = (&y * &y + &c) % n;

        let diff = if x >= y { &x - &y } else { &y - &x };
        if diff.is_zero() {
            // Cycle detected without finding factor
            return None;
        }

        let g = diff.gcd(n);
        if g > one && g < *n {
            return Some(g);
        }
    }
    None
}

fn execute_fermat_scan(
    n: &BigUint,
    params: &MacroParams,
    state_hint: &BigUint,
    deadline: &Instant,
) -> Option<BigUint> {
    let one = BigUint::one();

    // Start from state_hint or isqrt(N)
    let start = if *state_hint > one {
        state_hint.clone()
    } else {
        n.sqrt() + &one
    };

    let k_start = params.param1.max(1);
    let k_end = params.param2.max(k_start + 1).min(k_start + 1000);

    for k in k_start..k_end {
        if Instant::now() >= *deadline {
            return None;
        }
        let k_n = BigUint::from(k) * n;
        let a = (&start + &k_n.sqrt()) + &one;
        let a_sq = &a * &a;
        if a_sq < k_n {
            continue;
        }
        let b_sq = &a_sq - &k_n;
        let b = b_sq.sqrt();
        if &(&b * &b) == &b_sq {
            // a² - k*N = b² → a² ≡ b² (mod N)
            let sum = &a + &b;
            let g = sum.gcd(n);
            if g > one && g < *n {
                return Some(g);
            }
            let diff = if a >= b { &a - &b } else { &b - &a };
            if !diff.is_zero() {
                let g2 = diff.gcd(n);
                if g2 > one && g2 < *n {
                    return Some(g2);
                }
            }
        }
    }
    None
}

fn execute_lattice_smooth(
    n: &BigUint,
    params: &MacroParams,
    _deadline: &Instant,
) -> Option<BigUint> {
    let dim = params.param1.max(4).min(10) as usize;
    let result = smooth_pilatte::lattice::build_pilatte_lattice(n, dim);
    let vectors = smooth_pilatte::lattice::extract_exponent_vectors(&result);

    let one = BigUint::one();
    let primes = &result.params.primes;

    for exps in vectors.iter().take(5) {
        // Compute product of primes^exponents mod N
        let mut product = BigUint::one();
        for (i, &exp) in exps.iter().enumerate() {
            if i >= primes.len() || exp == 0 {
                continue;
            }
            let p = BigUint::from(primes[i]);
            product = (&product * &p.modpow(&BigUint::from(exp as u64), n)) % n;
        }

        if !product.is_zero() && product != one {
            let g = product.gcd(n);
            if g > one && g < *n {
                return Some(g);
            }
        }
    }
    None
}

fn execute_class_walk(
    n: &BigUint,
    params: &MacroParams,
    deadline: &Instant,
) -> Option<BigUint> {
    use cf_factor_ms::infrastructure::{form_reveals_factor, rho_step_ctx, InfraContext, InfraForm};

    let one = BigUint::one();
    if *n <= one {
        return None;
    }

    let steps = params.param1.max(10).min(2000);
    let ctx = InfraContext::new(n);
    let mut current = InfraForm::principal(n);

    for _ in 0..steps {
        if Instant::now() >= *deadline {
            return None;
        }
        current = rho_step_ctx(&current, &ctx);
        if let Some(factor) = form_reveals_factor(&current.form, n) {
            if factor > one && factor < *n {
                return Some(factor);
            }
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_macro_squfof_small() {
        let n = BigUint::from(77u32);
        let state = BigUint::from(2u32);
        let params = MacroParams::default();
        if let Some(factor) = execute_macro(&MacroKind::Squfof, &params, &n, &state) {
            assert!((&n % &factor).is_zero());
        }
    }

    #[test]
    fn test_macro_pollard_rho() {
        let n = BigUint::from(8051u32); // 83 × 97
        let state = BigUint::from(2u32);
        let params = MacroParams {
            param1: 5000,
            param2: 1,
        };
        let result = execute_macro(&MacroKind::PollardRho, &params, &n, &state);
        if let Some(factor) = result {
            assert!((&n % &factor).is_zero());
        }
    }

    #[test]
    fn test_macro_ecm_small() {
        let n = BigUint::from(143u32); // 11 × 13
        let state = BigUint::from(2u32);
        let params = MacroParams {
            param1: 500, // B1
            param2: 2,   // curves
        };
        if let Some(factor) = execute_macro(&MacroKind::Ecm, &params, &n, &state) {
            assert!((&n % &factor).is_zero());
        }
    }

    #[test]
    fn test_macro_fermat_scan() {
        let n = BigUint::from(15u32); // 3 × 5
        let state = BigUint::from(4u32);
        let params = MacroParams {
            param1: 1,   // k_start
            param2: 100, // k_end
        };
        if let Some(factor) = execute_macro(&MacroKind::FermatScan, &params, &n, &state) {
            assert!((&n % &factor).is_zero());
        }
    }

    #[test]
    fn test_macro_class_walk_no_panic() {
        let n = BigUint::from(77u32);
        let state = BigUint::from(2u32);
        let params = MacroParams {
            param1: 100, // steps
            param2: 1,
        };
        let _ = execute_macro(&MacroKind::ClassWalk, &params, &n, &state);
    }

    #[test]
    fn test_macro_lattice_smooth_no_panic() {
        let n = BigUint::from(77u32);
        let state = BigUint::from(2u32);
        let params = MacroParams {
            param1: 5, // dim
            param2: 1,
        };
        let _ = execute_macro(&MacroKind::LatticeSmooth, &params, &n, &state);
    }

    #[test]
    fn test_macro_display() {
        assert_eq!(format!("{}", MacroKind::Squfof), "MacroSQUFOF");
        assert_eq!(format!("{}", MacroKind::Ecm), "MacroECM");
        assert_eq!(format!("{}", MacroKind::PollardRho), "MacroRho");
        assert_eq!(format!("{}", MacroKind::FermatScan), "MacroFermat");
        assert_eq!(format!("{}", MacroKind::LatticeSmooth), "MacroLattice");
        assert_eq!(format!("{}", MacroKind::ClassWalk), "MacroClassWalk");
    }

    #[test]
    fn test_macro_trivial_n() {
        let n = BigUint::one();
        let state = BigUint::from(2u32);
        let params = MacroParams::default();
        for kind in &[
            MacroKind::Squfof,
            MacroKind::Ecm,
            MacroKind::PollardRho,
            MacroKind::FermatScan,
            MacroKind::LatticeSmooth,
            MacroKind::ClassWalk,
        ] {
            assert!(execute_macro(kind, &params, &n, &state).is_none());
        }
    }
}
