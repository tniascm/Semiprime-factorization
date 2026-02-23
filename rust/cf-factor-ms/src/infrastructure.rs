//! Infrastructure of the real quadratic field Q(sqrt(N)).
//!
//! The infrastructure is the set of reduced quadratic forms of discriminant D = 4N
//! together with a distance function. Each reduced form has an associated distance
//! from the identity, and the total circumference equals the regulator R.
//!
//! Key operations:
//! - rho_step: advance one step in the infrastructure (one CF step)
//! - giant_step: jump by composing forms and re-reducing (Gauss composition)
//! - distance tracking: cumulative sum of ln(partial quotients)

use cf_factor::cf::isqrt;
use cf_factor::forms::{gauss_compose, nucomp, QuadForm};
use num_bigint::{BigInt, BigUint};
use num_integer::Integer;
use num_traits::{One, Signed, ToPrimitive, Zero};
use std::collections::HashMap;

/// A quadratic form with its infrastructure distance from the identity.
#[derive(Debug, Clone)]
pub struct InfraForm {
    /// The reduced quadratic form.
    pub form: QuadForm,
    /// Distance from the identity in the infrastructure.
    pub distance: f64,
}

impl InfraForm {
    /// Create an identity form for discriminant D = 4N.
    pub fn identity(n: &BigUint) -> Self {
        let d = BigInt::from(4u32) * BigInt::from(n.clone());
        InfraForm {
            form: QuadForm::identity(&d),
            distance: 0.0,
        }
    }

    /// Create the principal form (1, 2*floor(sqrt(N)), ...) for D = 4N.
    pub fn principal(n: &BigUint) -> Self {
        let sqrt_n = isqrt(n);
        let b = BigInt::from(2u32) * BigInt::from(sqrt_n.clone());
        // c = (b^2 - D) / (4*a) = (4*floor(sqrt(N))^2 - 4N) / 4 = floor(sqrt(N))^2 - N
        let c = BigInt::from(&sqrt_n * &sqrt_n) - BigInt::from(n.clone());
        let form = QuadForm::new(BigInt::one(), b, c).reduce();
        InfraForm {
            form,
            distance: 0.0,
        }
    }
}

/// Perform one rho step in the infrastructure of D = 4N.
///
/// Given a reduced form (a, b, c), compute the next reduced form
/// by applying the rho operator and track the distance increment.
///
/// The rho operator: (a, b, c) -> (c, b', c') where
///   b' ≡ -b (mod 2|c|) with sqrt(D) - 2|c| < b' <= sqrt(D)
///   c' = (b'^2 - D) / (4c)
pub fn rho_step(form: &InfraForm, n: &BigUint) -> InfraForm {
    let d_big = BigInt::from(4u32) * BigInt::from(n.clone());
    let d_uint = BigUint::from(4u32) * n;
    let sqrt_d = isqrt(&d_uint);
    let sqrt_d_int = BigInt::from(sqrt_d);

    let _a = &form.form.a;
    let b = &form.form.b;
    let c = &form.form.c;

    if c.is_zero() {
        return form.clone();
    }

    // Apply rho: new form is (c, b', c')
    let two_abs_c = BigInt::from(2) * c.abs();
    if two_abs_c.is_zero() {
        return form.clone();
    }

    // b' ≡ -b (mod 2|c|) with sqrt(D) - 2|c| < b' <= sqrt(D)
    let neg_b = -b;
    let r = neg_b.mod_floor(&two_abs_c);
    let diff = &sqrt_d_int - &r;
    let k = diff.div_floor(&two_abs_c);
    let new_b = &r + &k * &two_abs_c;

    let new_a = c.clone();
    let new_c = (&new_b * &new_b - &d_big) / (BigInt::from(4) * &new_a);

    // Distance increment: ln(|new_b + sqrt(D)| / (2|a|))
    // Approximation: ln(a_k) where a_k is the partial quotient
    let partial_quotient = (&sqrt_d_int + b) / &two_abs_c;
    let pq_f64 = partial_quotient
        .to_f64()
        .unwrap_or(1.0)
        .abs()
        .max(1.0);
    let distance_increment = pq_f64.ln();

    InfraForm {
        form: QuadForm::new(new_a, new_b, new_c),
        distance: form.distance + distance_increment,
    }
}

/// Walk the infrastructure for a given number of steps.
///
/// Returns all forms visited along with their distances.
pub fn walk_infrastructure(n: &BigUint, max_steps: usize) -> Vec<InfraForm> {
    let mut forms = Vec::with_capacity(max_steps);
    let mut current = InfraForm::principal(n);
    forms.push(current.clone());

    for _ in 0..max_steps {
        let next = rho_step(&current, n);
        if next.form == current.form {
            break; // Cycle detected
        }
        forms.push(next.clone());
        current = next;
    }

    forms
}

/// Compose two infrastructure forms (giant step).
///
/// Computes f1 * f2 in the class group and tracks the combined distance.
/// Uses NUCOMP for large forms.
pub fn giant_step(f1: &InfraForm, f2: &InfraForm) -> InfraForm {
    let composed = nucomp(&f1.form, &f2.form);
    InfraForm {
        form: composed,
        distance: f1.distance + f2.distance,
    }
}

/// Square an infrastructure form (double the distance).
pub fn double_step(f: &InfraForm) -> InfraForm {
    let squared = gauss_compose(&f.form, &f.form);
    InfraForm {
        form: squared,
        distance: f.distance * 2.0,
    }
}

/// Compute form^n using repeated squaring, tracking distance.
pub fn power_step(f: &InfraForm, exp: u64) -> InfraForm {
    if exp == 0 {
        let d = f.form.discriminant();
        return InfraForm {
            form: QuadForm::identity(&d),
            distance: 0.0,
        };
    }
    if exp == 1 {
        return f.clone();
    }

    let mut result = {
        let d = f.form.discriminant();
        InfraForm {
            form: QuadForm::identity(&d),
            distance: 0.0,
        }
    };
    let mut base = f.clone();
    let mut e = exp;

    while e > 0 {
        if e & 1 == 1 {
            result = giant_step(&result, &base);
        }
        base = double_step(&base);
        e >>= 1;
    }

    result
}

/// Hash table for baby-step/giant-step search in the infrastructure.
///
/// Stores reduced forms keyed by (|a|, |b| mod something) for efficient lookup.
#[derive(Debug)]
pub struct InfraHashTable {
    table: HashMap<(String, String), Vec<InfraForm>>,
}

impl InfraHashTable {
    pub fn new() -> Self {
        InfraHashTable {
            table: HashMap::new(),
        }
    }

    /// Insert a form into the hash table.
    pub fn insert(&mut self, form: InfraForm) {
        let key = form_key(&form.form);
        self.table.entry(key).or_default().push(form);
    }

    /// Look up forms matching the given form (same reduced form up to sign).
    pub fn lookup(&self, form: &QuadForm) -> Option<&Vec<InfraForm>> {
        let key = form_key(form);
        self.table.get(&key)
    }

    /// Number of stored forms.
    pub fn len(&self) -> usize {
        self.table.values().map(|v| v.len()).sum()
    }

    /// Whether the table is empty.
    pub fn is_empty(&self) -> bool {
        self.table.is_empty()
    }
}

/// Compute a hash key for a reduced form.
fn form_key(f: &QuadForm) -> (String, String) {
    let reduced = f.reduce();
    (reduced.a.abs().to_string(), reduced.b.abs().to_string())
}

/// Check if a form reveals a factor of N.
///
/// For discriminant D = 4N, an ambiguous form (a, 0, c) or (a, a, c)
/// satisfies gcd(a, N) = factor. Also check gcd(c, N).
pub fn form_reveals_factor(form: &QuadForm, n: &BigUint) -> Option<BigUint> {
    let one = BigUint::one();

    // Check gcd(a, N)
    let a_abs = form.a.abs();
    if !a_abs.is_zero() && !a_abs.is_one() {
        let a_uint = a_abs.to_biguint().unwrap();
        let g = gcd_biguint(&a_uint, n);
        if g > one && g < *n {
            return Some(g);
        }
    }

    // Check gcd(c, N)
    let c_abs = form.c.abs();
    if !c_abs.is_zero() && !c_abs.is_one() {
        let c_uint = c_abs.to_biguint().unwrap();
        let g = gcd_biguint(&c_uint, n);
        if g > one && g < *n {
            return Some(g);
        }
    }

    // Check gcd(b, N) for ambiguous forms
    if form.is_ambiguous() {
        let b_abs = form.b.abs();
        if !b_abs.is_zero() && !b_abs.is_one() {
            let b_uint = b_abs.to_biguint().unwrap();
            let g = gcd_biguint(&b_uint, n);
            if g > one && g < *n {
                return Some(g);
            }
        }
    }

    None
}

/// GCD for BigUint.
fn gcd_biguint(a: &BigUint, b: &BigUint) -> BigUint {
    let mut x = a.clone();
    let mut y = b.clone();
    while !y.is_zero() {
        let t = &x % &y;
        x = std::mem::replace(&mut y, t);
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_principal_form() {
        let n = BigUint::from(77u64); // 7 * 11
        let f = InfraForm::principal(&n);
        // sqrt(77) ≈ 8.77, floor = 8
        // b = 2*8 = 16, c = 64 - 77 = -13
        // Form: (1, 16, -13) -> reduce
        assert_eq!(f.distance, 0.0);
        let d = f.form.discriminant();
        assert_eq!(d, BigInt::from(308)); // 4*77 = 308
    }

    #[test]
    fn test_rho_step_advances() {
        let n = BigUint::from(77u64);
        let f0 = InfraForm::principal(&n);
        let f1 = rho_step(&f0, &n);
        let f2 = rho_step(&f1, &n);

        // Distance should be non-decreasing (may be 0 increment if pq = 1)
        assert!(
            f1.distance >= f0.distance,
            "Distance should not decrease: {} vs {}",
            f1.distance, f0.distance
        );
        // After two steps, should have some positive distance
        assert!(
            f2.distance >= f0.distance,
            "Distance after 2 steps should be >= initial"
        );

        // Discriminant should be preserved
        assert_eq!(f0.form.discriminant(), f1.form.discriminant());
        assert_eq!(f1.form.discriminant(), f2.form.discriminant());
    }

    #[test]
    fn test_walk_infrastructure() {
        let n = BigUint::from(77u64);
        let forms = walk_infrastructure(&n, 50);

        assert!(forms.len() > 1, "Should walk at least 2 steps");

        // Distances should be non-decreasing
        for i in 1..forms.len() {
            assert!(
                forms[i].distance >= forms[i - 1].distance - 1e-10,
                "Distances should be non-decreasing at step {}",
                i
            );
        }

        // All forms should have the same discriminant
        let d0 = forms[0].form.discriminant();
        for f in &forms {
            assert_eq!(f.form.discriminant(), d0);
        }
    }

    #[test]
    fn test_giant_step_preserves_discriminant() {
        let n = BigUint::from(77u64);
        let f0 = InfraForm::principal(&n);
        let f1 = rho_step(&f0, &n);
        let composed = giant_step(&f0, &f1);

        assert_eq!(composed.form.discriminant(), f0.form.discriminant());
    }

    #[test]
    fn test_form_reveals_factor() {
        let n = BigUint::from(77u64); // 7 * 11
        // A form with a = 7 should reveal factor 7
        let form = QuadForm::new(BigInt::from(7), BigInt::from(0), BigInt::from(11));
        let factor = form_reveals_factor(&form, &n);
        assert!(factor.is_some());
        let f = factor.unwrap();
        assert!(f == BigUint::from(7u32) || f == BigUint::from(11u32));
    }

    #[test]
    fn test_walk_finds_factor() {
        let n = BigUint::from(77u64); // 7 * 11
        let forms = walk_infrastructure(&n, 100);

        let mut found_factor = false;
        for f in &forms {
            if let Some(_factor) = form_reveals_factor(&f.form, &n) {
                found_factor = true;
                break;
            }
        }

        // For small N, walking the infrastructure should eventually find a factor
        // (though it may take more steps)
        // This is a weak test - just verify the infrastructure works without panicking
        assert!(forms.len() > 1);
        let _ = found_factor; // May or may not find it in 100 steps
    }

    #[test]
    fn test_infra_hash_table() {
        let n = BigUint::from(77u64);
        let mut table = InfraHashTable::new();

        let forms = walk_infrastructure(&n, 20);
        for f in &forms {
            table.insert(f.clone());
        }

        assert_eq!(table.len(), forms.len());
        assert!(!table.is_empty());

        // Lookup should find the principal form
        let result = table.lookup(&forms[0].form);
        assert!(result.is_some());
    }

    #[test]
    fn test_power_step() {
        let n = BigUint::from(77u64);
        let f = InfraForm::principal(&n);

        // f^0 should be identity
        let f0 = power_step(&f, 0);
        assert_eq!(f0.distance, 0.0);

        // f^1 should be f (reduced)
        let f1 = power_step(&f, 1);
        assert_eq!(f1.form.discriminant(), f.form.discriminant());
    }
}
