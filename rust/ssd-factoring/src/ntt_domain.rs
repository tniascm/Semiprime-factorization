//! Strategy 2: NTT (Number Theoretic Transform) Domain
//!
//! Transform the bit representation of N into the frequency domain where
//! convolution becomes pointwise multiplication. The idea is that the
//! sequential accumulation of bit contributions (a linear recurrence) might
//! be expressible as a convolution that can be computed in O(k log k) via NTT.
//!
//! In practice, for modular reduction, this is more of an exploratory probe
//! than a practical speedup. The NTT itself costs O(k log k), and we still
//! need per-divisor work. The value is in testing whether the SSD formulation
//! yields correct results when the intermediate domain is a transform space.

use rayon::prelude::*;

use crate::SsdFormulation;

/// NTT Domain strategy.
pub struct NttDomain {
    /// NTT modulus (prime of form k * 2^m + 1).
    pub modulus: u64,
}

impl NttDomain {
    /// Create with the standard NTT-friendly prime 998244353 = 119 * 2^23 + 1.
    pub fn new_default() -> Self {
        Self {
            modulus: 998244353,
        }
    }
}

/// Modular exponentiation: base^exp mod modulus.
pub fn mod_pow_u64(mut base: u64, mut exp: u64, modulus: u64) -> u64 {
    if modulus == 1 {
        return 0;
    }
    let mut result: u128 = 1;
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

/// Find a primitive root (generator) of Z/pZ for the given prime modulus.
///
/// For 998244353, the primitive root is 3, but we compute it generally.
pub fn find_primitive_root(modulus: u64) -> u64 {
    if modulus == 2 {
        return 1;
    }
    let phi = modulus - 1;

    // Factor phi to find all prime factors
    let mut factors: Vec<u64> = Vec::new();
    let mut n = phi;
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

    // Find generator by checking candidates
    for g in 2..modulus {
        let mut is_generator = true;
        for &f in &factors {
            if mod_pow_u64(g, phi / f, modulus) == 1 {
                is_generator = false;
                break;
            }
        }
        if is_generator {
            return g;
        }
    }
    panic!("No primitive root found for modulus {}", modulus);
}

/// In-place forward Number Theoretic Transform.
///
/// `a` must have length that is a power of 2.
/// `root` must be a primitive n-th root of unity modulo `modulus`.
pub fn ntt_forward(a: &mut [u64], modulus: u64, root: u64) {
    let n = a.len();
    assert!(n.is_power_of_two(), "NTT length must be a power of 2");
    if n == 1 {
        return;
    }

    // Bit-reversal permutation
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            a.swap(i, j);
        }
    }

    // Cooley-Tukey butterfly
    let m = modulus as u128;
    let mut len = 2;
    while len <= n {
        let half = len / 2;
        let w = mod_pow_u64(root, (n / len) as u64, modulus);
        let mut i = 0;
        while i < n {
            let mut wj: u128 = 1;
            for jj in 0..half {
                let u = a[i + jj] as u128;
                let v = (wj * a[i + jj + half] as u128) % m;
                a[i + jj] = ((u + v) % m) as u64;
                a[i + jj + half] = ((u + m - v) % m) as u64;
                wj = (wj * w as u128) % m;
            }
            i += len;
        }
        len <<= 1;
    }
}

/// In-place inverse Number Theoretic Transform.
///
/// `a` must have length that is a power of 2.
/// `root` must be the same primitive n-th root of unity used in the forward transform.
pub fn ntt_inverse(a: &mut [u64], modulus: u64, root: u64) {
    let n = a.len();
    assert!(n.is_power_of_two(), "NTT length must be a power of 2");

    // Inverse NTT = forward NTT with inverse root, then divide by n
    let inv_root = mod_pow_u64(root, modulus - 2, modulus);
    ntt_forward(a, modulus, inv_root);

    let inv_n = mod_pow_u64(n as u64, modulus - 2, modulus);
    let m = modulus as u128;
    for val in a.iter_mut() {
        *val = ((*val as u128 * inv_n as u128) % m) as u64;
    }
}

impl SsdFormulation for NttDomain {
    fn name(&self) -> &str {
        "NTT Domain"
    }

    fn dimensionality(&self, n: u64) -> usize {
        let bit_count = if n == 0 {
            1
        } else {
            (64 - n.leading_zeros()) as usize
        };
        // 2x length to avoid circular aliasing, rounded to power of 2
        (2 * bit_count).next_power_of_two()
    }

    fn parallel(&self, n: u64, divisors: &[u64]) -> Vec<u64> {
        let modulus = self.modulus;
        let bit_count = if n == 0 {
            1
        } else {
            (64 - n.leading_zeros()) as usize
        };
        // Use 2x length to avoid circular convolution aliasing, then round up to power of 2
        let ntt_len = (2 * bit_count).next_power_of_two();

        // Step 1: Represent n as bits, zero-padded to ntt_len
        let mut bits_ntt: Vec<u64> = (0..ntt_len)
            .map(|i| if i < bit_count { (n >> i) & 1 } else { 0 })
            .collect();

        // Step 2: Find root of unity for NTT of this length
        let g = find_primitive_root(modulus);
        let root = mod_pow_u64(g, (modulus - 1) / ntt_len as u64, modulus);

        // Step 3: Forward NTT of bits
        ntt_forward(&mut bits_ntt, modulus, root);

        // Step 4: For each divisor, compute N mod d via NTT-based convolution.
        //
        // We want the dot product: sum(bits[i] * weights[i]) where weights[i] = 2^i mod d.
        // NTT computes circular convolution: conv[k] = sum(a[i] * b[k-i mod n]).
        // So conv[0] = sum(a[i] * b[-i mod n]) = sum(a[i] * b[n-i mod n]).
        //
        // To get the dot product sum(a[i]*b[i]), we reverse the weights vector:
        // rev_weights[0] = weights[0], rev_weights[j] = weights[n-j] for j>0.
        // Then conv[0] = sum(a[i] * rev_weights[-i mod n]) = sum(a[i] * weights[i]).
        divisors
            .par_iter()
            .map(|&d| {
                if d <= 1 {
                    return 0;
                }

                // Build weight vector: weights[i] = 2^i mod d for i < bit_count, else 0
                let weights: Vec<u64> = (0..ntt_len)
                    .map(|i| {
                        if i < bit_count {
                            mod_pow_u64(2, i as u64, d)
                        } else {
                            0
                        }
                    })
                    .collect();

                // Reverse weights to convert convolution into dot product
                let mut rev_weights = vec![0u64; ntt_len];
                rev_weights[0] = weights[0];
                for j in 1..ntt_len {
                    rev_weights[j] = weights[ntt_len - j];
                }

                // Forward NTT of reversed weights
                ntt_forward(&mut rev_weights, modulus, root);

                // Pointwise multiply in NTT domain
                let m128 = modulus as u128;
                let mut product: Vec<u64> = bits_ntt
                    .iter()
                    .zip(rev_weights.iter())
                    .map(|(&b, &f)| ((b as u128 * f as u128) % m128) as u64)
                    .collect();

                // Inverse NTT
                ntt_inverse(&mut product, modulus, root);

                // The dot product is at index 0 of the convolution result
                product[0] % d
            })
            .collect()
    }
}
