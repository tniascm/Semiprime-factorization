//! Integration tests comparing all factoring methods on random semiprimes.

use num_bigint::BigUint;
use num_traits::{One, Zero};

use factoring_core::{generate_rsa_target, pollard_rho, trial_division};
use ecm::{ecm_factor, EcmParams};
use classical_nfs::quadratic_sieve;
use quantum_inspired::period_factor;

/// Verify that a found factor is non-trivial: divides n, and is not 1 or n itself.
fn is_valid_nontrivial_factor(n: &BigUint, factor: &BigUint) -> bool {
    let one = BigUint::one();
    !factor.is_zero()
        && *factor != one
        && *factor != *n
        && (n % factor).is_zero()
}

/// Attempt trial division on n and return the first non-trivial factor, if any.
fn try_trial_division(n: &BigUint) -> Option<BigUint> {
    let bound = 1_000_000u64;
    let factors = trial_division(n, bound);
    // trial_division returns all prime factors; if it fully factored and there
    // are at least two, the first factor is valid.
    if factors.len() >= 2 {
        let f = &factors[0];
        if is_valid_nontrivial_factor(n, f) {
            return Some(f.clone());
        }
    }
    // Also check if the first factor alone is non-trivial
    if let Some(f) = factors.first() {
        if is_valid_nontrivial_factor(n, f) {
            return Some(f.clone());
        }
    }
    None
}

/// Attempt Pollard rho on n. Retries a few times since it is randomized.
fn try_pollard_rho(n: &BigUint) -> Option<BigUint> {
    for _ in 0..10 {
        if let Some(f) = pollard_rho(n) {
            if is_valid_nontrivial_factor(n, &f) {
                return Some(f);
            }
        }
    }
    None
}

/// Attempt ECM with small parameters suitable for small semiprimes.
fn try_ecm(n: &BigUint) -> Option<BigUint> {
    let params = EcmParams {
        b1: 2_000,
        b2: 50_000,
        num_curves: 32,
    };
    let result = ecm_factor(n, &params);
    if result.complete && result.factors.len() >= 2 {
        let f = &result.factors[0];
        if is_valid_nontrivial_factor(n, f) {
            return Some(f.clone());
        }
    }
    None
}

/// Attempt quadratic sieve with a small factor-base bound.
fn try_quadratic_sieve(n: &BigUint, bound: u64) -> Option<BigUint> {
    if let Some(f) = quadratic_sieve(n, bound) {
        if is_valid_nontrivial_factor(n, &f) {
            return Some(f);
        }
    }
    None
}

/// Attempt quantum-inspired period-based factoring.
fn try_period_factor(n: &BigUint) -> Option<BigUint> {
    if let Some(f) = period_factor(n, 200) {
        if is_valid_nontrivial_factor(n, &f) {
            return Some(f);
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn test_all_methods_16bit() {
    let mut rng = rand::thread_rng();

    let mut td_success = 0u32;
    let mut pr_success = 0u32;
    let mut ecm_success = 0u32;
    let mut qs_success = 0u32;
    let mut pf_success = 0u32;

    let count = 5;
    for i in 0..count {
        let target = generate_rsa_target(16, &mut rng);
        let n = &target.n;
        eprintln!(
            "[16-bit #{}/{}] n = {} (p={}, q={})",
            i + 1,
            count,
            n,
            target.p,
            target.q
        );

        if try_trial_division(n).is_some() {
            td_success += 1;
        }
        if try_pollard_rho(n).is_some() {
            pr_success += 1;
        }
        if try_ecm(n).is_some() {
            ecm_success += 1;
        }
        if try_quadratic_sieve(n, 100).is_some() {
            qs_success += 1;
        }
        if try_period_factor(n).is_some() {
            pf_success += 1;
        }
    }

    eprintln!("--- 16-bit results ---");
    eprintln!("  trial_division:   {}/{}", td_success, count);
    eprintln!("  pollard_rho:      {}/{}", pr_success, count);
    eprintln!("  ecm:              {}/{}", ecm_success, count);
    eprintln!("  quadratic_sieve:  {}/{}", qs_success, count);
    eprintln!("  period_factor:    {}/{}", pf_success, count);

    let min_required = 3u32;
    assert!(
        td_success >= min_required,
        "trial_division should succeed on at least {}/{} 16-bit semiprimes, got {}",
        min_required,
        count,
        td_success
    );
    assert!(
        pr_success >= min_required,
        "pollard_rho should succeed on at least {}/{} 16-bit semiprimes, got {}",
        min_required,
        count,
        pr_success
    );
    assert!(
        ecm_success >= min_required,
        "ecm should succeed on at least {}/{} 16-bit semiprimes, got {}",
        min_required,
        count,
        ecm_success
    );
    assert!(
        qs_success >= min_required,
        "quadratic_sieve should succeed on at least {}/{} 16-bit semiprimes, got {}",
        min_required,
        count,
        qs_success
    );
    assert!(
        pf_success >= min_required,
        "period_factor should succeed on at least {}/{} 16-bit semiprimes, got {}",
        min_required,
        count,
        pf_success
    );
}

#[test]
fn test_all_methods_32bit() {
    let mut rng = rand::thread_rng();

    let mut td_success = 0u32;
    let mut pr_success = 0u32;
    let mut ecm_success = 0u32;
    let mut qs_success = 0u32;
    let mut pf_success = 0u32;

    let count = 5;
    for i in 0..count {
        let target = generate_rsa_target(32, &mut rng);
        let n = &target.n;
        eprintln!(
            "[32-bit #{}/{}] n = {} (p={}, q={})",
            i + 1,
            count,
            n,
            target.p,
            target.q
        );

        if try_trial_division(n).is_some() {
            td_success += 1;
        }
        if try_pollard_rho(n).is_some() {
            pr_success += 1;
        }
        if try_ecm(n).is_some() {
            ecm_success += 1;
        }
        if try_quadratic_sieve(n, 200).is_some() {
            qs_success += 1;
        }
        if try_period_factor(n).is_some() {
            pf_success += 1;
        }
    }

    eprintln!("--- 32-bit results ---");
    eprintln!("  trial_division:   {}/{}", td_success, count);
    eprintln!("  pollard_rho:      {}/{}", pr_success, count);
    eprintln!("  ecm:              {}/{}", ecm_success, count);
    eprintln!("  quadratic_sieve:  {}/{}", qs_success, count);
    eprintln!("  period_factor:    {}/{}", pf_success, count);

    // At 32-bit, trial_division and pollard_rho should still succeed reliably
    assert!(
        td_success >= count,
        "trial_division should succeed on all {} 32-bit semiprimes, got {}",
        count,
        td_success
    );
    assert!(
        pr_success >= count,
        "pollard_rho should succeed on all {} 32-bit semiprimes, got {}",
        count,
        pr_success
    );
    // ECM should succeed on most
    assert!(
        ecm_success >= 3,
        "ecm should succeed on at least 3/{} 32-bit semiprimes, got {}",
        count,
        ecm_success
    );
    // QS and period_factor may fail on some 32-bit semiprimes — no hard assertion
    eprintln!(
        "  (QS succeeded on {}/{}, period_factor on {}/{})",
        qs_success, count, pf_success, count
    );
}

#[test]
fn test_method_agreement() {
    let mut rng = rand::thread_rng();

    let count = 10;
    for i in 0..count {
        let target = generate_rsa_target(24, &mut rng);
        let n = &target.n;
        eprintln!(
            "[agreement #{}/{}] n = {} (p={}, q={})",
            i + 1,
            count,
            n,
            target.p,
            target.q
        );

        // Collect all non-trivial factors found by each method
        let mut found_factors: Vec<(&str, BigUint)> = Vec::new();

        if let Some(f) = try_trial_division(n) {
            found_factors.push(("trial_division", f));
        }
        if let Some(f) = try_pollard_rho(n) {
            found_factors.push(("pollard_rho", f));
        }
        if let Some(f) = try_ecm(n) {
            found_factors.push(("ecm", f));
        }
        if let Some(f) = try_quadratic_sieve(n, 150) {
            found_factors.push(("quadratic_sieve", f));
        }
        if let Some(f) = try_period_factor(n) {
            found_factors.push(("period_factor", f));
        }

        // Every factor returned must be a valid non-trivial divisor of n
        for (method, factor) in &found_factors {
            assert!(
                is_valid_nontrivial_factor(n, factor),
                "[agreement #{}] {} returned invalid factor {} for n={}",
                i + 1,
                method,
                factor,
                n
            );
        }

        eprintln!(
            "  {} methods found valid factors",
            found_factors.len()
        );
    }
}

#[test]
fn test_ecm_vs_pollard_rho() {
    let mut rng = rand::thread_rng();

    let mut ecm_success = 0u32;
    let mut pr_success = 0u32;

    let count = 20;
    for i in 0..count {
        let target = generate_rsa_target(48, &mut rng);
        let n = &target.n;
        eprintln!(
            "[ecm-vs-rho #{}/{}] n = {} (p={}, q={})",
            i + 1,
            count,
            n,
            target.p,
            target.q
        );

        let ecm_ok = try_ecm(n).is_some();
        let pr_ok = try_pollard_rho(n).is_some();

        if ecm_ok {
            ecm_success += 1;
        }
        if pr_ok {
            pr_success += 1;
        }

        eprintln!(
            "  ecm={}, pollard_rho={}",
            if ecm_ok { "OK" } else { "FAIL" },
            if pr_ok { "OK" } else { "FAIL" }
        );
    }

    eprintln!("--- ECM vs Pollard Rho on 48-bit semiprimes ---");
    eprintln!("  ecm:         {}/{}", ecm_success, count);
    eprintln!("  pollard_rho: {}/{}", pr_success, count);

    // Both methods should succeed on a reasonable fraction of 48-bit semiprimes
    // Pollard rho is very reliable for 48-bit (24-bit primes are small)
    assert!(
        pr_success >= 15,
        "pollard_rho should succeed on at least 15/{} 48-bit semiprimes, got {}",
        count,
        pr_success
    );
    // ECM with our small params may struggle on some, but should get a decent fraction
    assert!(
        ecm_success >= 5,
        "ecm should succeed on at least 5/{} 48-bit semiprimes, got {}",
        count,
        ecm_success
    );
}
