//! Dimension formulas for spaces of cusp forms S_k(Gamma_0(N)).
//!
//! The key formula for weight 2:
//!   dim S_2(Gamma_0(N)) = 1 + index/12 - nu2/4 - nu3/3 - nu_inf/2
//! where:
//!   index = N * product_{p|N} (1 + 1/p)  (index of Gamma_0(N) in SL_2(Z))
//!   nu2 = product_{p^e || N} nu2_pe       (elliptic points of order 2)
//!   nu3 = product_{p^e || N} nu3_pe       (elliptic points of order 3)
//!   nu_inf = number of cusps

use num_integer::Integer;

/// Compute the Legendre symbol (a/p) for odd prime p.
/// Returns 1 if a is a quadratic residue mod p, -1 if not, 0 if p | a.
pub fn legendre_symbol(a: i64, p: u64) -> i64 {
    assert!(p > 2, "Legendre symbol requires odd prime");
    let p_i = p as i64;
    let a_mod = ((a % p_i) + p_i) % p_i;
    if a_mod == 0 {
        return 0;
    }
    // Euler's criterion: (a/p) = a^((p-1)/2) mod p
    let exp = (p - 1) / 2;
    let result = mod_pow(a_mod as u64, exp, p);
    if result == 1 {
        1
    } else if result == p - 1 {
        -1
    } else {
        // Should not happen for prime p
        0
    }
}

/// Compute the Kronecker symbol (a/n), extending the Jacobi symbol.
pub fn kronecker_symbol(a: i64, n: u64) -> i64 {
    if n == 0 {
        if a.unsigned_abs() == 1 {
            return 1;
        } else {
            return 0;
        }
    }
    if n == 1 {
        return 1;
    }
    if n == 2 {
        let a_mod8 = ((a % 8) + 8) % 8;
        return match a_mod8 {
            0 => 0,
            1 | 7 => 1,
            3 | 5 => -1,
            _ => 0,
        };
    }

    // Factor out powers of 2 from n
    let mut result = 1i64;
    let mut remaining = n;

    let twos = remaining.trailing_zeros();
    if twos > 0 {
        remaining >>= twos;
        // (a/2)^twos
        let a2 = kronecker_symbol(a, 2);
        if twos % 2 == 1 {
            result *= a2;
        }
    }

    // Now remaining is odd
    if remaining == 1 {
        return result;
    }

    // Factor remaining into primes and compute Legendre symbols
    let factors = factor_u64(remaining);
    for (p, e) in &factors {
        let ls = legendre_symbol(a, *p);
        if *e % 2 == 1 {
            result *= ls;
        }
        // Even powers contribute ls^(2k) = 1 if ls != 0, else 0
        if ls == 0 {
            return 0;
        }
    }
    result
}

/// Modular exponentiation: base^exp mod modulus.
pub fn mod_pow(mut base: u64, mut exp: u64, modulus: u64) -> u64 {
    if modulus == 1 {
        return 0;
    }
    let mut result = 1u128;
    let m = modulus as u128;
    base %= modulus;
    let mut b = base as u128;
    while exp > 0 {
        if exp & 1 == 1 {
            result = (result * b) % m;
        }
        exp >>= 1;
        b = (b * b) % m;
    }
    result as u64
}

/// Factor a u64 into (prime, exponent) pairs.
pub fn factor_u64(mut n: u64) -> Vec<(u64, u32)> {
    let mut factors = Vec::new();
    if n <= 1 {
        return factors;
    }
    let mut d = 2u64;
    while d * d <= n {
        if n % d == 0 {
            let mut e = 0u32;
            while n % d == 0 {
                n /= d;
                e += 1;
            }
            factors.push((d, e));
        }
        d += 1;
    }
    if n > 1 {
        factors.push((n, 1));
    }
    factors
}

/// Get distinct prime factors of n.
pub fn prime_factors(n: u64) -> Vec<u64> {
    factor_u64(n).into_iter().map(|(p, _)| p).collect()
}

/// Get all divisors of n.
pub fn divisors(n: u64) -> Vec<u64> {
    let mut divs = Vec::new();
    let mut d = 1u64;
    while d * d <= n {
        if n % d == 0 {
            divs.push(d);
            if d != n / d {
                divs.push(n / d);
            }
        }
        d += 1;
    }
    divs.sort();
    divs
}

/// Euler's totient function phi(n).
pub fn euler_phi(n: u64) -> u64 {
    if n <= 1 {
        return n;
    }
    let factors = factor_u64(n);
    let mut result = n;
    for (p, _) in &factors {
        result = result / p * (p - 1);
    }
    result
}

/// Count elliptic points of order 2 for Gamma_0(N).
/// nu_2(N) = product_{p^e || N} nu_2(p^e)
/// where nu_2(p^e) = 0 if e >= 2, (1 + (-1/p)) if e = 1, and 1 if p=2,e=1 gives 0.
/// More precisely: nu_2(N) = product_{p^e || N} (1 + Kronecker(-1, p)) if all e=1,
/// and 0 if any e >= 2 (for the standard formula).
///
/// Actually the general formula:
///   nu_2(p^e) = 0 if e >= 2
///   nu_2(p) = 1 + Legendre(-1/p) for odd p
///   nu_2(2) = 0
/// Then nu_2(N) = product over p^e || N of nu_2(p^e).
pub fn count_elliptic_2(n: u64) -> u64 {
    if n <= 1 {
        return 1;
    }
    let factors = factor_u64(n);
    let mut result = 1u64;
    for (p, e) in &factors {
        if *e >= 2 {
            return 0;
        }
        if *p == 2 {
            // (1 + Kronecker(-1, 2)) = 1 + 1 = 2? No.
            // For p=2: Legendre symbol is not defined, use Kronecker.
            // (-1/2) = 1 (since -1 = 7 mod 8, and (*/2) = 1 for * = +/-1 mod 8)
            // Actually: Kronecker(-1, 2) = (-1)^((2^2-1)/8) ... no.
            // By definition: (-1 mod 8) = 7. (a/2) = 0 if a even, 1 if a=+/-1 mod 8, -1 if a=+/-3 mod 8.
            // -1 mod 8 = 7 which is -1 mod 8, so (-1/2) = 1.
            // So nu_2(2) = 1 + 1 = 2? But the standard reference says nu_2(2) = 0.
            //
            // The correct formula from Shimura/Diamond-Shurman:
            // nu_2(N) = product_{p|N} (1 + (-4/p)) where (-4/p) is the Kronecker symbol.
            // (-4/2) = 0, so nu_2 = 0 whenever 2 | N.
            // Actually: for squarefree N, nu_2(N) = product_{p|N} (1 + chi_{-4}(p))
            // where chi_{-4} is the Kronecker symbol (-4/.).
            // chi_{-4}(2) = 0, so if 2|N then nu_2(N) = 0.
            return 0;
        }
        // For odd prime p with e=1: 1 + Kronecker(-4, p) = 1 + Legendre(-1, p)
        let chi = legendre_symbol(-1, *p);
        let factor = (1 + chi) as u64;
        result *= factor;
    }
    result
}

/// Count elliptic points of order 3 for Gamma_0(N).
/// nu_3(N) = product_{p^e || N} of local factors.
/// For squarefree N: nu_3(N) = product_{p|N} (1 + chi_{-3}(p))
/// where chi_{-3} = Kronecker(-3, .).
/// chi_{-3}(3) = 0, so if 3|N then nu_3 = 0.
pub fn count_elliptic_3(n: u64) -> u64 {
    if n <= 1 {
        return 1;
    }
    let factors = factor_u64(n);
    let mut result = 1u64;
    for (p, e) in &factors {
        if *e >= 2 {
            return 0;
        }
        if *p == 2 {
            // Kronecker(-3, 2): -3 mod 8 = 5. (a/2) = -1 for a = +/-3 mod 8.
            // So (-3/2) = -1. Factor = 1 + (-1) = 0.
            // When 2 | N, nu_3 = 0 if this local factor is 0.
            // Actually: -3 mod 8 = 5. The Kronecker symbol (a/2) for odd a:
            // = 1 if a = +/-1 mod 8 (i.e., a mod 8 in {1, 7})
            // = -1 if a = +/-3 mod 8 (i.e., a mod 8 in {3, 5})
            // -3 mod 8 = 5, which is in {3, 5}, so (-3/2) = -1.
            // Factor = 1 + (-1) = 0.
            return 0;
        } else if *p == 3 {
            // Kronecker(-3, 3) = 0, so factor = 1 + 0 = 1.
            // 3 dividing N makes the local factor 1, product continues.
            result *= 1;
        } else {
            let chi = legendre_symbol(-3, *p);
            let factor = (1 + chi) as u64;
            result *= factor;
        }
    }
    result
}

/// Count cusps of Gamma_0(N).
/// c(N) = sum_{d | N} phi(gcd(d, N/d))
pub fn count_cusps(n: u64) -> u64 {
    let divs = divisors(n);
    let mut sum = 0u64;
    for d in &divs {
        let nd = n / d;
        let g = d.gcd(&nd);
        sum += euler_phi(g);
    }
    sum
}

/// Index of Gamma_0(N) in SL_2(Z).
/// mu(N) = N * product_{p | N} (1 + 1/p)
/// This equals sum_{d|N} phi(d) = psi(N).
pub fn psi_index(n: u64) -> u64 {
    if n == 0 {
        return 0;
    }
    let factors = prime_factors(n);
    // mu = N * product (1 + 1/p) = N * product (p+1)/p
    // To compute exactly in integers: start with N, multiply by (p+1), divide by p.
    // Since p | N, the division is exact at each step (if we process in order).
    let mut result = n;
    for p in &factors {
        result = result / p * (p + 1);
    }
    result
}

/// Dimension of S_2(Gamma_0(N)) for N >= 1.
///
/// Formula (for N squarefree or general):
///   g = 1 + mu/12 - nu2/4 - nu3/3 - c/2
/// where mu = psi(N) (index), nu2, nu3 are elliptic point counts, c = cusps.
///
/// This gives the genus of X_0(N), and dim S_2(Gamma_0(N)) = g (genus formula).
pub fn dim_s2(n: u64) -> u64 {
    if n <= 1 {
        return 0;
    }

    let mu = psi_index(n);
    let nu2 = count_elliptic_2(n);
    let nu3 = count_elliptic_3(n);
    let c = count_cusps(n);

    // genus = 1 + mu/12 - nu2/4 - nu3/3 - c/2
    // Multiply everything by 12 to stay in integers:
    // 12*g = 12 + mu - 3*nu2 - 4*nu3 - 6*c
    let twelve_g = 12i64 + mu as i64 - 3 * nu2 as i64 - 4 * nu3 as i64 - 6 * c as i64;

    // g should be >= 0 and twelve_g should be divisible by 12
    if twelve_g < 0 {
        return 0;
    }
    (twelve_g / 12) as u64
}

/// Dimension of the new subspace S_2^new(Gamma_0(N)).
///
/// dim S_2^new(N) = sum_{d | N} mu(N/d) * dim S_2(Gamma_0(d))
/// where mu is the Mobius function.
///
/// Equivalently: dim S_2(N) = sum_{d | N} sigma_0(N/d) * dim S_2^new(d)
/// where sigma_0 counts divisors. So by Mobius inversion on the divisor lattice:
/// dim S_2^new(N) = sum_{d | N} mu(N/d) * dim S_2(d)
/// (where mu here is the standard Mobius function, not the index).
///
/// Wait, the correct formula is:
/// dim S_2(N) = sum_{M | N} sum_{d | (N/M)} dim S_2^new(M)
///            = sum_{M | N} sigma_0(N/M) * dim S_2^new(M)
///
/// So by Mobius inversion (multiplicative):
/// dim S_2^new(N) = sum_{d | N} mu(d) * dim S_2(N/d)
/// Hmm, this doesn't invert sigma_0 correctly. Let me use the correct inversion.
///
/// Actually the old subspace for Gamma_0(N) is:
/// S_2^old(N) = sum over proper divisors M of N, sum over d | (N/M):
///   images of S_2^new(M) under the maps f(z) -> f(dz)
///
/// The dimension formula is:
/// dim S_2(N) = sum_{M | N} tau(N/M) * dim S_2^new(M)
/// where tau(k) = number of divisors of k.
///
/// Mobius inversion of f = tau * g gives g(N) = sum_{d|N} lambda(d) * f(N/d)
/// where lambda is the Liouville-like inverse of tau under Dirichlet convolution.
///
/// The inverse of tau under Dirichlet convolution is the function
/// tau^{-1}(n) = sum_{d|n} mu(d) * mu(n/d) ... no that's mu*mu.
///
/// Actually: if f = g * tau (Dirichlet convolution), then
/// g = f * tau^{-1}. And tau^{-1} = mu * mu (Dirichlet convolution of mu with itself),
/// since tau = 1 * 1 and tau^{-1} = mu * mu.
///
/// So: dim S_2^new(N) = sum_{d | N} (mu * mu)(d) * dim S_2(N/d)
///                     = sum_{d | N} sum_{e | d} mu(e) * mu(d/e) * dim S_2(N/d)
pub fn dim_s2_new(n: u64) -> u64 {
    let divs = divisors(n);

    // Compute (mu * mu)(d) for each divisor d of N
    // (mu * mu)(d) = sum_{e | d} mu(e) * mu(d/e)
    let mut result = 0i64;
    for d in &divs {
        let mu_star_mu = dirichlet_mu_mu(*d);
        let dim = dim_s2(n / d) as i64;
        result += mu_star_mu * dim;
    }

    // Result should be non-negative
    result.max(0) as u64
}

/// Dimension of the old subspace.
pub fn dim_s2_old(n: u64) -> u64 {
    let total = dim_s2(n);
    let new = dim_s2_new(n);
    total.saturating_sub(new)
}

/// Standard Mobius function mu(n).
pub fn mobius(n: u64) -> i64 {
    if n == 1 {
        return 1;
    }
    let factors = factor_u64(n);
    for (_, e) in &factors {
        if *e >= 2 {
            return 0;
        }
    }
    if factors.len() % 2 == 0 {
        1
    } else {
        -1
    }
}

/// (mu * mu)(n) = sum_{d | n} mu(d) * mu(n/d), Dirichlet convolution of mu with itself.
fn dirichlet_mu_mu(n: u64) -> i64 {
    let divs = divisors(n);
    let mut sum = 0i64;
    for d in &divs {
        sum += mobius(*d) * mobius(n / d);
    }
    sum
}

/// Check if n is prime using trial division.
pub fn is_prime(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    if n == 2 || n == 3 {
        return true;
    }
    if n % 2 == 0 || n % 3 == 0 {
        return false;
    }
    let mut d = 5u64;
    while d * d <= n {
        if n % d == 0 || n % (d + 2) == 0 {
            return false;
        }
        d += 6;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_legendre_symbol() {
        // (1/p) = 1 for all p
        assert_eq!(legendre_symbol(1, 3), 1);
        assert_eq!(legendre_symbol(1, 5), 1);
        assert_eq!(legendre_symbol(1, 7), 1);

        // (-1/p): 1 if p = 1 mod 4, -1 if p = 3 mod 4
        assert_eq!(legendre_symbol(-1, 3), -1); // 3 = 3 mod 4
        assert_eq!(legendre_symbol(-1, 5), 1);  // 5 = 1 mod 4
        assert_eq!(legendre_symbol(-1, 7), -1); // 7 = 3 mod 4
        assert_eq!(legendre_symbol(-1, 13), 1); // 13 = 1 mod 4

        // (2/p): 1 if p = +/-1 mod 8, -1 if p = +/-3 mod 8
        assert_eq!(legendre_symbol(2, 7), 1);   // 7 = -1 mod 8
        assert_eq!(legendre_symbol(2, 3), -1);  // 3 = 3 mod 8
        assert_eq!(legendre_symbol(2, 5), -1);  // 5 = -3 mod 8
        assert_eq!(legendre_symbol(2, 17), 1);  // 17 = 1 mod 8
    }

    #[test]
    fn test_dim_s2_primes() {
        // Known dimensions of S_2(Gamma_0(p)) for small primes:
        // p=2: 0, p=3: 0, p=5: 0, p=7: 0, p=11: 1, p=13: 0
        // p=17: 1, p=19: 1, p=23: 2, p=29: 2, p=31: 2, p=37: 2
        // p=41: 3, p=43: 3, p=47: 4, p=53: 4, p=59: 4, p=61: 4
        assert_eq!(dim_s2(2), 0);
        assert_eq!(dim_s2(3), 0);
        assert_eq!(dim_s2(5), 0);
        assert_eq!(dim_s2(7), 0);
        assert_eq!(dim_s2(11), 1);
        assert_eq!(dim_s2(13), 0);
        assert_eq!(dim_s2(17), 1);
        assert_eq!(dim_s2(19), 1);
        assert_eq!(dim_s2(23), 2);
        assert_eq!(dim_s2(29), 2);
        assert_eq!(dim_s2(31), 2);
        assert_eq!(dim_s2(37), 2);
        assert_eq!(dim_s2(41), 3);
        assert_eq!(dim_s2(43), 3);
        assert_eq!(dim_s2(47), 4);
    }

    #[test]
    fn test_dim_s2_composites() {
        // N = 77 = 7 * 11: dim S_2(77) should be 7
        // Can verify: genus of X_0(77)
        // mu(77) = 77*(1+1/7)*(1+1/11) = 77 * 8/7 * 12/11 = 77*96/77 = 96
        // nu2(77) = (1+(-1/7))*(1+(-1/11)) = (1+(-1))*(1+(+1)) = 0*2 = 0
        //   (-1/7): 7=3mod4 => -1; (-1/11): 11=3mod4 => -1
        //   so (1-1)*(1-1) = 0
        // nu3(77) = (1+(-3/7))*(1+(-3/11))
        //   (-3/7): need to check. -3 mod 7 = 4. (4/7) = 4^3 = 64 = 1 mod 7 => 1.
        //   So 1+1 = 2.
        //   (-3/11): -3 mod 11 = 8. (8/11) = 8^5 = 32768. 32768 mod 11 = 32768-2978*11 = 32768-32758=10 = -1 mod 11. So -1.
        //   So 1+(-1) = 0. nu3 = 2*0 = 0.
        // c(77): divisors of 77: 1,7,11,77.
        //   phi(gcd(1,77))=phi(1)=1
        //   phi(gcd(7,11))=phi(1)=1
        //   phi(gcd(11,7))=phi(1)=1
        //   phi(gcd(77,1))=phi(1)=1
        //   c = 4
        // g = 1 + 96/12 - 0/4 - 0/3 - 4/2 = 1 + 8 - 0 - 0 - 2 = 7
        assert_eq!(dim_s2(77), 7);
    }

    #[test]
    fn test_dim_s2_new_composites() {
        // For N = pq (squarefree): dim_new(pq) = dim(pq) - 2*dim(p) - 2*dim(q)
        // (since dim(1)=0, and each of level p, q lifts twice)
        //
        // N=77: dim(77)=7, dim(7)=0, dim(11)=1
        // dim_new(77) = 7 - 2*0 - 2*1 = 5
        assert_eq!(dim_s2_new(77), 5);

        // N=143 = 11*13: dim(143)
        // mu = 143*(12/11)*(14/13) = 143*168/143 = 168
        // nu2: (-1/11)=-1 => 1-1=0. So nu2=0.
        // nu3: (-3/11)=-1 => 0. So nu3=0.
        // c: divisors 1,11,13,143. phi(gcd(1,143))+phi(gcd(11,13))+phi(gcd(13,11))+phi(gcd(143,1))
        //  = 1+1+1+1 = 4
        // g = 1 + 168/12 - 0 - 0 - 4/2 = 1 + 14 - 2 = 13
        // dim_new = 13 - 2*dim(11) - 2*dim(13) = 13 - 2*1 - 2*0 = 11
        assert_eq!(dim_s2(143), 13);
        assert_eq!(dim_s2_new(143), 11);
    }

    #[test]
    fn test_euler_phi() {
        assert_eq!(euler_phi(1), 1);
        assert_eq!(euler_phi(2), 1);
        assert_eq!(euler_phi(6), 2);
        assert_eq!(euler_phi(12), 4);
    }

    #[test]
    fn test_count_cusps() {
        // For prime p: cusps = 2 (divisors 1 and p, gcd(1,p)=1 and gcd(p,1)=1, phi(1)=1 each)
        assert_eq!(count_cusps(11), 2);
        assert_eq!(count_cusps(7), 2);
        // For p*q: 4 cusps
        assert_eq!(count_cusps(77), 4);
    }

    #[test]
    fn test_mobius() {
        assert_eq!(mobius(1), 1);
        assert_eq!(mobius(2), -1);
        assert_eq!(mobius(3), -1);
        assert_eq!(mobius(4), 0); // 4 = 2^2
        assert_eq!(mobius(6), 1); // 6 = 2*3, two prime factors
        assert_eq!(mobius(30), -1); // 30 = 2*3*5, three prime factors
    }
}
