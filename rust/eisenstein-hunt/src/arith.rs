/// Modular arithmetic primitives for u64 with u128 intermediates.

use rand::Rng;

/// Modular exponentiation: base^exp mod m using binary method.
pub fn mod_pow(mut base: u64, mut exp: u64, m: u64) -> u64 {
    if m == 1 {
        return 0;
    }
    let mut result = 1u128;
    let m = m as u128;
    base %= m as u64;
    let mut b = base as u128;
    while exp > 0 {
        if exp & 1 == 1 {
            result = result * b % m;
        }
        exp >>= 1;
        b = b * b % m;
    }
    result as u64
}

/// Extended GCD: returns (gcd, x, y) such that a*x + b*y = gcd.
fn extended_gcd(a: i128, b: i128) -> (i128, i128, i128) {
    if a == 0 {
        return (b, 0, 1);
    }
    let (g, x, y) = extended_gcd(b % a, a);
    (g, y - (b / a) * x, x)
}

/// Modular inverse: a^{-1} mod m. Returns None if gcd(a,m) != 1.
pub fn mod_inv(a: u64, m: u64) -> Option<u64> {
    let (g, x, _) = extended_gcd(a as i128, m as i128);
    if g != 1 {
        return None;
    }
    Some(((x % m as i128 + m as i128) % m as i128) as u64)
}

/// Jacobi symbol (a/n) for odd n > 0.
pub fn jacobi_symbol(mut a: i64, mut n: u64) -> i64 {
    if n == 1 {
        return 1;
    }
    if a == 0 {
        return 0;
    }
    // Normalize a into [0, n)
    a = ((a % n as i64) + n as i64) as i64 % n as i64;
    let mut a = a as u64;
    let mut result = 1i64;

    while a != 0 {
        // Extract factors of 2
        while a % 2 == 0 {
            a /= 2;
            let n_mod8 = n % 8;
            if n_mod8 == 3 || n_mod8 == 5 {
                result = -result;
            }
        }
        // Quadratic reciprocity
        std::mem::swap(&mut a, &mut n);
        if a % 4 == 3 && n % 4 == 3 {
            result = -result;
        }
        a %= n;
    }

    if n == 1 {
        result
    } else {
        0
    }
}

/// Kronecker symbol (a/n), extending Jacobi to all n.
pub fn kronecker_symbol(a: i64, n: u64) -> i64 {
    if n == 0 {
        return if a.unsigned_abs() == 1 { 1 } else { 0 };
    }
    if n == 1 {
        return 1;
    }

    // Factor out powers of 2 from n
    let mut n_odd = n;
    let mut twos = 0u32;
    while n_odd % 2 == 0 {
        n_odd /= 2;
        twos += 1;
    }

    let mut result = 1i64;
    if twos > 0 {
        // (a/2) = 0 if a is even, 1 if a ≡ ±1 (mod 8), -1 if a ≡ ±3 (mod 8)
        let a_mod = ((a % 8) + 8) % 8;
        let kr2 = match a_mod {
            0 | 2 | 4 | 6 => 0i64,
            1 | 7 => 1,
            3 | 5 => -1,
            _ => unreachable!(),
        };
        if kr2 == 0 {
            return 0;
        }
        if twos % 2 == 1 {
            result *= kr2;
        }
    }

    if n_odd == 1 {
        return result;
    }

    result * jacobi_symbol(a, n_odd)
}

/// Lucas V_n(P, Q) mod m via double-and-add binary chain.
/// Doubling: V_{2k} = V_k^2 - 2*Q^k, V_{2k+1} = V_k*V_{k+1} - P*Q^k.
/// Increment: V_{k+1} = P*V_k - Q*V_{k-1} (encoded via the (vk, vk1) pair).
pub fn lucas_v(n: u64, p: u64, q: u64, m: u64) -> u64 {
    if m == 1 {
        return 0;
    }
    if n == 0 {
        return 2 % m;
    }
    if n == 1 {
        return p % m;
    }

    let m128 = m as u128;
    let p128 = p as u128 % m128;
    let q128 = q as u128 % m128;

    // Track (V_k, V_{k+1}, Q^k) starting at k=0
    let mut vk: u128 = 2;      // V_0
    let mut vk1: u128 = p128;  // V_1
    let mut qk: u128 = 1;      // Q^0

    let bits = 64 - n.leading_zeros();
    for i in (0..bits).rev() {
        // Double: k -> 2k
        let vk1_new = (vk * vk1 % m128 + m128 - p128 * qk % m128) % m128;
        let vk_new = (vk * vk % m128 + 2 * m128 - 2 * (qk % m128) % m128) % m128;
        let qk_new = qk * qk % m128;

        vk = vk_new;
        vk1 = vk1_new;
        qk = qk_new;

        if (n >> i) & 1 == 1 {
            // Increment: 2k -> 2k+1
            // V_{2k+1} is already in vk1
            // V_{2k+2} = P * V_{2k+1} - Q * V_{2k}
            let vk_inc = vk1;
            let vk1_inc = (p128 * vk1 % m128 + m128 - q128 * vk % m128) % m128;
            let qk_inc = q128 * qk % m128;

            vk = vk_inc;
            vk1 = vk1_inc;
            qk = qk_inc;
        }
    }

    vk as u64
}

/// Lucas U_n(P, Q) mod m via binary chain.
/// Uses V and U relationship: U_{2k} = U_k * V_k, U_{2k+1} = U_{k+1}*V_k - Q^k.
/// Simpler approach: track (U_k, V_k, Q^k) together.
pub fn lucas_u(n: u64, p: u64, q: u64, m: u64) -> u64 {
    if m == 1 {
        return 0;
    }
    if n == 0 {
        return 0;
    }
    if n == 1 {
        return 1 % m;
    }

    let m128 = m as u128;
    let p128 = p as u128 % m128;
    let q128 = q as u128 % m128;

    // Discriminant D = P^2 - 4Q
    // Use identity: 2*U_n = (D*U_n + ...); actually simpler to track U and V together
    // U_{2k} = U_k * V_k
    // V_{2k} = V_k^2 - 2*Q^k
    // U_{2k+1} = (P * U_{2k} + V_{2k}) / 2 ... this requires division
    // Better: use matrix method [[P, -Q], [1, 0]]^n applied to [1, 0]
    // [[P, -Q], [1, 0]]^n = [[U_{n+1}, -Q*U_n], [U_n, -Q*U_{n-1}]] (up to sign)
    // Just do matrix exponentiation.

    // Matrix [[a, b], [c, d]] represents the state
    let mut a = p128;
    let mut b = (m128 - q128) % m128; // -Q mod m
    let mut c = 1u128;
    let mut d = 0u128;

    // Result matrix (identity)
    let mut ra = 1u128;
    let mut rb = 0u128;
    let mut rc = 0u128;
    let mut rd = 1u128;

    let mut exp = n - 1;
    while exp > 0 {
        if exp & 1 == 1 {
            // result = result * mat
            let na = (ra * a % m128 + rb * c % m128) % m128;
            let nb = (ra * b % m128 + rb * d % m128) % m128;
            let nc = (rc * a % m128 + rd * c % m128) % m128;
            let nd = (rc * b % m128 + rd * d % m128) % m128;
            ra = na;
            rb = nb;
            rc = nc;
            rd = nd;
        }
        exp >>= 1;
        // mat = mat * mat
        let na = (a * a % m128 + b * c % m128) % m128;
        let nb = (a * b % m128 + b * d % m128) % m128;
        let nc = (c * a % m128 + d * c % m128) % m128;
        let nd = (c * b % m128 + d * d % m128) % m128;
        a = na;
        b = nb;
        c = nc;
        d = nd;
    }

    // [[ra, rb], [rc, rd]] * [1, 0]^T = [ra, rc]
    // U_n = rc (bottom-left after multiplying [[P,-Q],[1,0]]^{n-1} by [U_1, U_0] = [1, 0])
    // Actually: M^{n-1} applied to initial vector [U_1, U_0] = [1, 0] gives [U_n, U_{n-1}]
    // So U_n = ra * 1 + rb * 0 = ra
    ra as u64
}

/// Fermat quotient: (N^{ℓ-1} - 1) / ℓ mod ℓ.
/// Requires ℓ not dividing N.
pub fn fermat_quotient(n: u64, ell: u64) -> Option<u64> {
    if n % ell == 0 {
        return None;
    }
    let ell_sq = (ell as u128) * (ell as u128);
    if ell_sq > u64::MAX as u128 {
        return None; // ℓ too large for u64 modulus
    }
    let ell_sq = ell_sq as u64;
    let pow = mod_pow(n % ell_sq, ell - 1, ell_sq);
    // pow ≡ 1 (mod ℓ) by Fermat, so (pow - 1) is divisible by ℓ
    let diff = if pow >= 1 { pow - 1 } else { ell_sq + pow - 1 };
    Some((diff / ell) % ell)
}

/// Binomial coefficient C(n, j) mod ℓ using multiplicative formula in F_ℓ.
pub fn binomial_mod(n: u64, j: u64, ell: u64) -> u64 {
    if j == 0 {
        return 1 % ell;
    }
    let n_mod = n % ell;
    if n_mod < j {
        return 0;
    }

    let mut num = 1u128;
    let mut den = 1u128;
    let m = ell as u128;

    for i in 0..j {
        num = num * ((n_mod - i) as u128) % m;
        den = den * ((i + 1) as u128) % m;
    }

    let den_inv = mod_inv(den as u64, ell).unwrap_or(0);
    (num as u64 as u128 * den_inv as u128 % m) as u64
}

/// Find a primitive root modulo ℓ (ℓ prime).
pub fn primitive_root(ell: u64) -> u64 {
    if ell == 2 {
        return 1;
    }
    // Factor ℓ-1
    let mut factors = Vec::new();
    let mut n = ell - 1;
    let mut d = 2u64;
    while d * d <= n {
        if n % d == 0 {
            factors.push(d);
            while n % d == 0 {
                n /= d;
            }
        }
        d += 1;
    }
    if n > 1 {
        factors.push(n);
    }

    // Test candidates
    for g in 2..ell {
        let is_root = factors.iter().all(|&f| {
            mod_pow(g, (ell - 1) / f, ell) != 1
        });
        if is_root {
            return g;
        }
    }
    unreachable!("prime must have a primitive root")
}

/// Discrete logarithm: find x such that g^x ≡ a (mod ℓ), or None.
/// Brute force — fine for ℓ ≤ 43867.
pub fn discrete_log(a: u64, g: u64, ell: u64) -> Option<u64> {
    if a % ell == 0 {
        return None;
    }
    let target = a % ell;
    let mut power = 1u64;
    for x in 0..ell - 1 {
        if power == target {
            return Some(x);
        }
        power = (power as u128 * g as u128 % ell as u128) as u64;
    }
    None
}

/// Deterministic Miller-Rabin primality test for n < 3.3 * 10^24.
/// Uses witnesses {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37}.
pub fn is_prime_u64(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    if n < 4 {
        return true;
    }
    if n % 2 == 0 || n % 3 == 0 {
        return false;
    }

    // Write n-1 = 2^s * d
    let mut d = n - 1;
    let mut s = 0u32;
    while d % 2 == 0 {
        d /= 2;
        s += 1;
    }

    let witnesses = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37];
    'outer: for &a in &witnesses {
        if a >= n {
            continue;
        }
        let mut x = mod_pow(a, d, n);
        if x == 1 || x == n - 1 {
            continue;
        }
        for _ in 0..s - 1 {
            x = (x as u128 * x as u128 % n as u128) as u64;
            if x == n - 1 {
                continue 'outer;
            }
        }
        return false;
    }
    true
}

/// Generate a random prime with exactly `bits` bits (top bit set).
pub fn random_prime(bits: u32, rng: &mut impl Rng) -> u64 {
    assert!(bits >= 2 && bits <= 63);
    let lo = 1u64 << (bits - 1);
    let hi = if bits == 63 { u64::MAX } else { (1u64 << bits) - 1 };
    loop {
        let mut n = rng.gen_range(lo..=hi);
        n |= 1; // ensure odd
        n |= lo; // ensure top bit
        if is_prime_u64(n) {
            return n;
        }
    }
}

/// Continued fraction convergents of a/b. Returns up to `max_terms` pairs (h_i, k_i).
pub fn cf_convergents(a: u64, b: u64, max_terms: usize) -> Vec<(u64, u64)> {
    let mut convergents = Vec::new();
    let mut a = a;
    let mut b = b;
    let mut h_prev = 0u128;
    let mut h_curr = 1u128;
    let mut k_prev = 1u128;
    let mut k_curr = 0u128;

    for _ in 0..max_terms {
        if b == 0 {
            break;
        }
        let q = a / b;
        let r = a % b;

        let h_next = q as u128 * h_curr + h_prev;
        let k_next = q as u128 * k_curr + k_prev;

        convergents.push((h_next as u64, k_next as u64));

        h_prev = h_curr;
        h_curr = h_next;
        k_prev = k_curr;
        k_curr = k_next;

        a = b;
        b = r;
    }

    convergents
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mod_pow_basic() {
        assert_eq!(mod_pow(2, 10, 1000), 24);
        assert_eq!(mod_pow(3, 0, 7), 1);
        assert_eq!(mod_pow(5, 690, 691), 1); // Fermat's little theorem
        assert_eq!(mod_pow(0, 5, 7), 0);
        assert_eq!(mod_pow(7, 1, 7), 0);
    }

    #[test]
    fn test_mod_pow_large() {
        // 2^16 = 65536; 65536 mod 691 = 65536 - 94*691 = 65536 - 64954 = 582
        assert_eq!(mod_pow(2, 16, 691), 65536 % 691);
    }

    #[test]
    fn test_mod_inv() {
        assert_eq!(mod_inv(3, 7), Some(5)); // 3*5 = 15 ≡ 1 (mod 7)
        assert_eq!(mod_inv(2, 691), Some(346)); // 2*346 = 692 ≡ 1 (mod 691)
        assert_eq!(mod_inv(0, 7), None);
        assert_eq!(mod_inv(6, 9), None); // gcd(6,9) = 3
    }

    #[test]
    fn test_jacobi_symbol() {
        // (1/n) = 1 for all n
        assert_eq!(jacobi_symbol(1, 3), 1);
        assert_eq!(jacobi_symbol(1, 691), 1);

        // (2/p): 2 is QR mod p iff p ≡ ±1 (mod 8)
        // 691 mod 8 = 3, so (2/691) = -1
        assert_eq!(jacobi_symbol(2, 691), -1);
        // 7 mod 8 = 7, so (2/7) = 1
        assert_eq!(jacobi_symbol(2, 7), 1);

        // (-1/p) = 1 iff p ≡ 1 (mod 4)
        // 691 mod 4 = 3, so (-1/691) = -1
        assert_eq!(jacobi_symbol(-1, 691), -1);
        // 5 mod 4 = 1, so (-1/5) = 1
        assert_eq!(jacobi_symbol(-1, 5), 1);
    }

    #[test]
    fn test_kronecker_symbol() {
        // Kronecker should agree with Jacobi for odd primes
        assert_eq!(kronecker_symbol(2, 691), jacobi_symbol(2, 691));
        assert_eq!(kronecker_symbol(-1, 5), jacobi_symbol(-1, 5));

        // (a/1) = 1
        assert_eq!(kronecker_symbol(42, 1), 1);
    }

    #[test]
    fn test_lucas_v_basic() {
        // V_0(P, Q) = 2
        assert_eq!(lucas_v(0, 3, 1, 100), 2);
        // V_1(P, Q) = P
        assert_eq!(lucas_v(1, 7, 1, 100), 7);
        // V_2(P, Q) = P^2 - 2Q
        assert_eq!(lucas_v(2, 5, 1, 100), 23); // 25-2=23
        // V_3(P, Q) = P^3 - 3PQ
        assert_eq!(lucas_v(3, 5, 1, 1000), 110); // 125-15=110
    }

    #[test]
    fn test_lucas_u_basic() {
        // U_0 = 0, U_1 = 1
        assert_eq!(lucas_u(0, 3, 1, 100), 0);
        assert_eq!(lucas_u(1, 3, 1, 100), 1);
        // U_2(P,Q) = P
        assert_eq!(lucas_u(2, 5, 1, 100), 5);
        // U_3(P,Q) = P^2 - Q
        assert_eq!(lucas_u(3, 5, 1, 100), 24); // 25-1=24
    }

    #[test]
    fn test_fermat_quotient() {
        let fq = fermat_quotient(2, 691).unwrap();
        assert!(fq < 691);
        // Verify: 2^690 ≡ 1 + 691*fq (mod 691^2)
        let pow = mod_pow(2, 690, 691 * 691);
        assert_eq!((pow - 1) / 691, fq);
    }

    #[test]
    fn test_binomial_mod() {
        // C(5, 2) = 10
        assert_eq!(binomial_mod(5, 2, 691), 10);
        // C(10, 3) = 120
        assert_eq!(binomial_mod(10, 3, 691), 120);
        // C(n, 0) = 1
        assert_eq!(binomial_mod(100, 0, 691), 1);
        // C(n, n) requires n_mod >= j check
        assert_eq!(binomial_mod(5, 5, 691), 1);
    }

    #[test]
    fn test_primitive_root() {
        let g = primitive_root(691);
        // g should generate all of F_691*
        assert_eq!(mod_pow(g, 690, 691), 1);
        // g^{690/2} should NOT be 1
        assert_ne!(mod_pow(g, 345, 691), 1);
    }

    #[test]
    fn test_discrete_log() {
        let g = primitive_root(691);
        let a = mod_pow(g, 42, 691);
        assert_eq!(discrete_log(a, g, 691), Some(42));
    }

    #[test]
    fn test_is_prime() {
        assert!(is_prime_u64(2));
        assert!(is_prime_u64(3));
        assert!(!is_prime_u64(4));
        assert!(is_prime_u64(691));
        assert!(is_prime_u64(3617));
        assert!(is_prime_u64(43867));
        assert!(!is_prime_u64(15));
        assert!(!is_prime_u64(1));
    }

    #[test]
    fn test_random_prime() {
        let mut rng = rand::thread_rng();
        for bits in [8, 12, 16] {
            let p = random_prime(bits, &mut rng);
            assert!(is_prime_u64(p));
            assert!(p >= 1u64 << (bits - 1));
            assert!(p < 1u64 << bits);
        }
    }

    #[test]
    fn test_cf_convergents() {
        // CF(355/113) = [3; 7, 16, ...]
        // convergents: 3/1, 22/7, 355/113
        let convs = cf_convergents(355, 113, 5);
        assert_eq!(convs[0], (3, 1));
        assert_eq!(convs[1], (22, 7));
        assert_eq!(convs[2], (355, 113));
    }
}
