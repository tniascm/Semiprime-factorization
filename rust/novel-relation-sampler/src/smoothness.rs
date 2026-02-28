//! Smoothness checking via trial division.

/// Trial-divide val by primes in the factor base.
/// Returns (exponents, cofactor). If cofactor == 1, val is smooth.
pub fn trial_divide(mut val: u64, factor_base: &[u64]) -> (Vec<u32>, u64) {
    let mut exponents = vec![0u32; factor_base.len()];
    for (i, &p) in factor_base.iter().enumerate() {
        if p == 0 {
            continue;
        }
        while val % p == 0 {
            val /= p;
            exponents[i] += 1;
        }
        if val == 1 {
            break;
        }
    }
    (exponents, val)
}

/// Check if a value is B-smooth using the given factor base.
pub fn is_smooth(val: u64, factor_base: &[u64]) -> bool {
    let (_, cofactor) = trial_divide(val, factor_base);
    cofactor == 1
}

/// GCD for u64.
pub fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trial_divide_smooth() {
        let fb = vec![2, 3, 5, 7];
        let (exps, cofactor) = trial_divide(360, &fb);
        assert_eq!(exps, vec![3, 2, 1, 0]);
        assert_eq!(cofactor, 1);
    }

    #[test]
    fn test_trial_divide_not_smooth() {
        let fb = vec![2, 3, 5];
        let (_, cofactor) = trial_divide(77, &fb);
        assert_eq!(cofactor, 77);
    }

    #[test]
    fn test_is_smooth() {
        let fb = vec![2, 3, 5, 7];
        assert!(is_smooth(360, &fb));
        assert!(!is_smooth(77, &fb));
        assert!(is_smooth(1, &fb));
    }

    #[test]
    fn test_gcd() {
        assert_eq!(gcd(12, 8), 4);
        assert_eq!(gcd(7, 13), 1);
        assert_eq!(gcd(0, 5), 5);
    }
}
