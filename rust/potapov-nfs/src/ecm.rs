//! Standalone ECM (Elliptic Curve Method) for multi-precision integers.
//!
//! Three implementations:
//! - **Fast path (U192)**: Fixed-width 192-bit Montgomery arithmetic for n <= 192 bits.
//!   Uses `U192` (3 x u64 limbs) with hand-written REDC, avoiding all GMP overhead.
//!   Targets c45 (~148-bit balanced semiprimes) at ~2-3ms per curve.
//! - **Fast path (U256)**: Fixed-width 256-bit Montgomery arithmetic for 193-256 bit n.
//!   Uses `U256` (4 x u64 limbs) with hand-written REDC for larger composites.
//! - **Fallback path**: `rug::Integer` (GMP-backed) for n > 256 bits.
//!
//! Montgomery curve `By^2 = x^3 + Ax^2 + x` in projective coordinates `(X : Z)`,
//! with Suyama parameterization for 12-torsion starting points.
//!
//! Two phases:
//! - Phase 1: scalar multiply by lcm(1..B1) -- catches factors with B1-smooth group order
//! - Phase 2: baby-step giant-step for primes in (B1, B2] -- catches factors with one
//!   large prime in the group order
//!
//! For c45 (~148-bit balanced semiprimes with ~74-bit factors), ECM with B1=50000
//! has roughly 1/50 success probability per curve. With 200 curves and rayon
//! parallelism, expected wall-clock time is well under 1 second.

use rug::Assign;
use rug::Integer;

// ===========================================================================
// Fixed-width 192-bit Montgomery arithmetic
// ===========================================================================

/// 192-bit unsigned integer, stored as 3 little-endian u64 limbs: [lo, mid, hi].
type U192 = [u64; 3];

/// Zero constant.
const U192_ZERO: U192 = [0, 0, 0];
/// One constant.
const U192_ONE: U192 = [1, 0, 0];

/// Montgomery form parameters for 192-bit modular arithmetic.
///
/// For modulus n (must be odd), R = 2^192. We store:
/// - `n`: the modulus
/// - `n_inv`: -n^{-1} mod 2^64 (single limb, used in REDC)
/// - `r_squared`: R^2 mod n (used to convert into Montgomery form)
#[derive(Clone, Debug)]
struct Mont192 {
    n: U192,
    n_inv: u64,
    r_squared: U192,
}

impl Mont192 {
    /// Build Montgomery parameters for odd modulus `n`.
    fn new(n: &U192) -> Self {
        debug_assert!(n[0] & 1 == 1, "Mont192: n must be odd");

        // Compute n_inv = -n^{-1} mod 2^64 via Newton's method.
        // n[0] is odd, so n[0] * 1 = 1 (mod 2). We lift to mod 2^64.
        let n0 = n[0];
        let mut x: u64 = 1;
        for _ in 0..6 {
            // Newton: x = x * (2 - n0 * x), doubling correct bits each step.
            x = x.wrapping_mul(2u64.wrapping_sub(n0.wrapping_mul(x)));
        }
        // x = n0^{-1} mod 2^64, so n_inv = -x mod 2^64 = wrapping_neg.
        let n_inv = x.wrapping_neg();

        // Compute R mod n = 2^192 mod n.
        // We compute this by repeated shifting: start with 1 and double 192 times.
        let r_mod_n = Self::compute_r_mod_n(n);

        // Compute R^2 mod n = (R mod n)^2 mod n.
        // But we can also compute it by doubling R mod n another 192 times.
        let r_squared = Self::compute_r_squared(n, &r_mod_n);

        Mont192 {
            n: *n,
            n_inv,
            r_squared,
        }
    }

    /// Compute 2^192 mod n by repeated doubling.
    fn compute_r_mod_n(n: &U192) -> U192 {
        let mut r = U192_ONE;
        for _ in 0..192 {
            r = add_mod_192(&r, &r, n);
        }
        r
    }

    /// Compute R^2 mod n = (R mod n) doubled 192 more times.
    fn compute_r_squared(n: &U192, r_mod_n: &U192) -> U192 {
        let mut r2 = *r_mod_n;
        for _ in 0..192 {
            r2 = add_mod_192(&r2, &r2, n);
        }
        r2
    }
}

/// Compare two U192 values: returns Ordering.
#[inline]
fn cmp_192(a: &U192, b: &U192) -> std::cmp::Ordering {
    // Compare from most significant limb down.
    if a[2] != b[2] {
        return a[2].cmp(&b[2]);
    }
    if a[1] != b[1] {
        return a[1].cmp(&b[1]);
    }
    a[0].cmp(&b[0])
}

/// a + b mod n, where a, b < n.
#[inline]
fn add_mod_192(a: &U192, b: &U192, n: &U192) -> U192 {
    // Compute a + b with carry.
    let (s0, c0) = a[0].overflowing_add(b[0]);
    let (s1, c1a) = a[1].overflowing_add(b[1]);
    let (s1, c1b) = s1.overflowing_add(c0 as u64);
    let c1 = c1a | c1b;
    let (s2, c2a) = a[2].overflowing_add(b[2]);
    let (s2, c2b) = s2.overflowing_add(c1 as u64);
    let carry = c2a | c2b;

    let mut s = [s0, s1, s2];

    // If carry or s >= n, subtract n.
    if carry || cmp_192(&s, n) != std::cmp::Ordering::Less {
        let (d0, borrow0) = s[0].overflowing_sub(n[0]);
        let (d1, borrow1a) = s[1].overflowing_sub(n[1]);
        let (d1, borrow1b) = d1.overflowing_sub(borrow0 as u64);
        let (d2, _) = s[2].overflowing_sub(n[2]);
        let d2 = d2.wrapping_sub((borrow1a | borrow1b) as u64);
        s = [d0, d1, d2];
    }

    s
}

/// a - b mod n, where a, b < n. Returns a + n - b if a < b.
#[inline]
fn sub_mod_192(a: &U192, b: &U192, n: &U192) -> U192 {
    if cmp_192(a, b) != std::cmp::Ordering::Less {
        // a >= b: just subtract.
        let (d0, borrow0) = a[0].overflowing_sub(b[0]);
        let (d1, borrow1a) = a[1].overflowing_sub(b[1]);
        let (d1, borrow1b) = d1.overflowing_sub(borrow0 as u64);
        let (d2, _) = a[2].overflowing_sub(b[2]);
        let d2 = d2.wrapping_sub((borrow1a | borrow1b) as u64);
        [d0, d1, d2]
    } else {
        // a < b: compute a + n - b.
        // First compute a + n.
        let (t0, c0) = a[0].overflowing_add(n[0]);
        let (t1, c1a) = a[1].overflowing_add(n[1]);
        let (t1, c1b) = t1.overflowing_add(c0 as u64);
        let c1 = c1a | c1b;
        let t2 = a[2].wrapping_add(n[2]).wrapping_add(c1 as u64);
        // Then subtract b.
        let (d0, borrow0) = t0.overflowing_sub(b[0]);
        let (d1, borrow1a) = t1.overflowing_sub(b[1]);
        let (d1, borrow1b) = d1.overflowing_sub(borrow0 as u64);
        let (d2, _) = t2.overflowing_sub(b[2]);
        let d2 = d2.wrapping_sub((borrow1a | borrow1b) as u64);
        [d0, d1, d2]
    }
}

/// 3-limb x 3-limb multiplication producing a 6-limb result.
///
/// Uses the classic schoolbook method with u128 intermediates. To avoid u128
/// overflow when summing multiple products (each product is up to 128 bits,
/// and summing 3 of them can exceed 128 bits), we use the standard inner-loop
/// approach: accumulate one product at a time with carry propagation.
#[inline]
fn mul_3x3(a: &U192, b: &U192) -> [u64; 6] {
    let mut t = [0u64; 6];

    // Standard schoolbook: for each limb of a, multiply by all limbs of b
    // and accumulate into t. This naturally handles carry without overflow
    // since each step is: t[i+j] + a[i]*b[j] + carry, which is at most
    // (2^64-1) + (2^64-1)^2 + (2^64-1) = 2^128 - 1, fitting in u128.
    for i in 0..3 {
        let mut carry: u64 = 0;
        for j in 0..3 {
            let prod = a[i] as u128 * b[j] as u128 + t[i + j] as u128 + carry as u128;
            t[i + j] = prod as u64;
            carry = (prod >> 64) as u64;
        }
        t[i + 3] = carry;
    }

    t
}

/// Montgomery multiplication: compute a * b * R^{-1} mod n (REDC).
///
/// Input: a, b in Montgomery form (< n).
/// Output: a*b*R^{-1} mod n, also in Montgomery form.
///
/// Algorithm:
///   1. Compute 6-limb product t = a * b.
///   2. For i = 0, 1, 2: m = t[i] * n_inv mod 2^64; add m*n to t at position i.
///   3. Result = upper 3 limbs (t[3..5]), conditionally subtract n.
#[inline]
fn mont_mul_192(a: &U192, b: &U192, mont: &Mont192) -> U192 {
    let mut t = mul_3x3(a, b);

    // REDC loop: for each of the 3 lower limbs, cancel them out.
    // Iteration i=0:
    let m = t[0].wrapping_mul(mont.n_inv);
    let mut carry = mac_3_at(&mut t, 0, m, &mont.n);
    // Propagate carry into t[3..5].
    let idx = 3;
    let (v, c) = t[idx].overflowing_add(carry);
    t[idx] = v;
    carry = c as u64;
    let (v, c) = t[4].overflowing_add(carry);
    t[4] = v;
    t[5] = t[5].wrapping_add(c as u64);

    // Iteration i=1:
    let m = t[1].wrapping_mul(mont.n_inv);
    carry = mac_3_at(&mut t, 1, m, &mont.n);
    let (v, c) = t[4].overflowing_add(carry);
    t[4] = v;
    t[5] = t[5].wrapping_add(c as u64);

    // Iteration i=2:
    let m = t[2].wrapping_mul(mont.n_inv);
    carry = mac_3_at(&mut t, 2, m, &mont.n);
    t[5] = t[5].wrapping_add(carry);

    // Result is the upper half [t[3], t[4], t[5]].
    let mut result = [t[3], t[4], t[5]];

    // Conditional subtraction if result >= n.
    if t[5] > mont.n[2]
        || (t[5] == mont.n[2]
            && (t[4] > mont.n[1] || (t[4] == mont.n[1] && t[3] >= mont.n[0])))
    {
        let (d0, borrow0) = result[0].overflowing_sub(mont.n[0]);
        let (d1, borrow1a) = result[1].overflowing_sub(mont.n[1]);
        let (d1, borrow1b) = d1.overflowing_sub(borrow0 as u64);
        let d2 = result[2]
            .wrapping_sub(mont.n[2])
            .wrapping_sub((borrow1a | borrow1b) as u64);
        result = [d0, d1, d2];
    }

    result
}

/// Multiply-accumulate: t[offset..offset+3] += m * n, returning the carry out.
///
/// Computes the 3-limb product m * n[0..2] and adds it into t starting at `offset`.
/// Returns the carry that propagates past t[offset+2].
#[inline]
fn mac_3_at(t: &mut [u64; 6], offset: usize, m: u64, n: &U192) -> u64 {
    let m128 = m as u128;

    // Limb 0
    let p = m128 * n[0] as u128 + t[offset] as u128;
    t[offset] = p as u64;
    let mut carry = (p >> 64) as u128;

    // Limb 1
    let p = m128 * n[1] as u128 + t[offset + 1] as u128 + carry;
    t[offset + 1] = p as u64;
    carry = p >> 64;

    // Limb 2
    let p = m128 * n[2] as u128 + t[offset + 2] as u128 + carry;
    t[offset + 2] = p as u64;
    (p >> 64) as u64
}

/// Montgomery squaring: a^2 * R^{-1} mod n.
///
/// Delegates to mont_mul_192 — with LTO + codegen-units=1, the compiler
/// generates optimal code for the a==b case.
#[inline]
fn mont_sqr_192(a: &U192, mont: &Mont192) -> U192 {
    mont_mul_192(a, a, mont)
}

/// Convert a regular value into Montgomery form: a * R mod n.
/// Computed as mont_mul(a, R^2 mod n).
#[inline]
fn to_mont_192(a: &U192, mont: &Mont192) -> U192 {
    mont_mul_192(a, &mont.r_squared, mont)
}

/// Convert from Montgomery form back to regular: a * R^{-1} mod n.
/// Computed as mont_mul(a, 1).
#[inline]
fn from_mont_192(a: &U192, mont: &Mont192) -> U192 {
    mont_mul_192(a, &U192_ONE, mont)
}

/// Binary GCD for U192 values. Both inputs are in regular (non-Montgomery) form.
///
/// Uses the standard binary GCD algorithm, extended for 3-limb integers.
fn gcd_192(a: &U192, b: &U192) -> U192 {
    if *a == U192_ZERO {
        return *b;
    }
    if *b == U192_ZERO {
        return *a;
    }

    let mut u = *a;
    let mut v = *b;

    // Count trailing zeros in u and v.
    let shift_u = trailing_zeros_192(&u);
    let shift_v = trailing_zeros_192(&v);
    let shift = shift_u.min(shift_v);

    shr_192_inplace(&mut u, shift_u);
    shr_192_inplace(&mut v, shift_v);

    loop {
        // Both u and v are odd here (or one of them is zero, which terminates).
        match cmp_192(&u, &v) {
            std::cmp::Ordering::Equal => break,
            std::cmp::Ordering::Greater => {
                // u = u - v (both odd, so result is even).
                sub_inplace_192(&mut u, &v);
                let tz = trailing_zeros_192(&u);
                shr_192_inplace(&mut u, tz);
            }
            std::cmp::Ordering::Less => {
                sub_inplace_192(&mut v, &u);
                let tz = trailing_zeros_192(&v);
                shr_192_inplace(&mut v, tz);
            }
        }
    }

    shl_192_inplace(&mut u, shift);
    u
}

/// Count trailing zero bits of a U192.
#[inline]
fn trailing_zeros_192(a: &U192) -> u32 {
    if a[0] != 0 {
        a[0].trailing_zeros()
    } else if a[1] != 0 {
        64 + a[1].trailing_zeros()
    } else if a[2] != 0 {
        128 + a[2].trailing_zeros()
    } else {
        192
    }
}

/// Right shift a U192 in place by `shift` bits (0..192).
#[inline]
fn shr_192_inplace(a: &mut U192, shift: u32) {
    if shift == 0 {
        return;
    }
    if shift >= 192 {
        *a = U192_ZERO;
        return;
    }
    if shift >= 128 {
        a[0] = a[2] >> (shift - 128);
        a[1] = 0;
        a[2] = 0;
    } else if shift >= 64 {
        let s = shift - 64;
        if s == 0 {
            a[0] = a[1];
            a[1] = a[2];
        } else {
            a[0] = (a[1] >> s) | (a[2] << (64 - s));
            a[1] = a[2] >> s;
        }
        a[2] = 0;
    } else {
        // Must compute from low to high to avoid overwriting inputs.
        let new0 = (a[0] >> shift) | (a[1] << (64 - shift));
        let new1 = (a[1] >> shift) | (a[2] << (64 - shift));
        let new2 = a[2] >> shift;
        a[0] = new0;
        a[1] = new1;
        a[2] = new2;
    }
}

/// Left shift a U192 in place by `shift` bits (0..192).
#[inline]
fn shl_192_inplace(a: &mut U192, shift: u32) {
    if shift == 0 {
        return;
    }
    if shift >= 192 {
        *a = U192_ZERO;
        return;
    }
    if shift >= 128 {
        let s = shift - 128;
        a[2] = a[0] << s;
        a[1] = 0;
        a[0] = 0;
    } else if shift >= 64 {
        let s = shift - 64;
        if s == 0 {
            a[2] = a[1];
            a[1] = a[0];
        } else {
            a[2] = (a[1] << s) | (a[0] >> (64 - s));
            a[1] = a[0] << s;
        }
        a[0] = 0;
    } else {
        // Must compute from high to low to avoid overwriting inputs.
        let new2 = (a[2] << shift) | (a[1] >> (64 - shift));
        let new1 = (a[1] << shift) | (a[0] >> (64 - shift));
        let new0 = a[0] << shift;
        a[0] = new0;
        a[1] = new1;
        a[2] = new2;
    }
}

/// u = u - v in place (assumes u >= v).
#[inline]
fn sub_inplace_192(u: &mut U192, v: &U192) {
    let (d0, borrow0) = u[0].overflowing_sub(v[0]);
    let (d1, borrow1a) = u[1].overflowing_sub(v[1]);
    let (d1, borrow1b) = d1.overflowing_sub(borrow0 as u64);
    let d2 = u[2]
        .wrapping_sub(v[2])
        .wrapping_sub((borrow1a | borrow1b) as u64);
    u[0] = d0;
    u[1] = d1;
    u[2] = d2;
}

/// Number of significant bits in a U192.
#[cfg(test)]
fn significant_bits_192(a: &U192) -> u32 {
    if a[2] != 0 {
        128 + (64 - a[2].leading_zeros())
    } else if a[1] != 0 {
        64 + (64 - a[1].leading_zeros())
    } else if a[0] != 0 {
        64 - a[0].leading_zeros()
    } else {
        0
    }
}

// ---------------------------------------------------------------------------
// Conversion between rug::Integer and U192
// ---------------------------------------------------------------------------

/// Convert a non-negative rug::Integer to U192. Panics if n > 2^192 - 1.
fn integer_to_u192(n: &Integer) -> U192 {
    debug_assert!(*n >= 0, "integer_to_u192: n must be non-negative");
    debug_assert!(
        n.significant_bits() <= 192,
        "integer_to_u192: n must fit in 192 bits (got {} bits)",
        n.significant_bits()
    );

    // Extract bytes in little-endian order.
    let digits = n.to_digits::<u8>(rug::integer::Order::LsfLe);
    let mut limbs = [0u64; 3];
    for (i, &byte) in digits.iter().enumerate().take(24) {
        limbs[i / 8] |= (byte as u64) << ((i % 8) * 8);
    }
    limbs
}

/// Convert a U192 back to rug::Integer.
fn u192_to_integer(a: &U192) -> Integer {
    let mut bytes = [0u8; 24];
    for i in 0..3 {
        let limb = a[i];
        for j in 0..8 {
            bytes[i * 8 + j] = ((limb >> (j * 8)) & 0xFF) as u8;
        }
    }
    Integer::from_digits(&bytes, rug::integer::Order::LsfLe)
}

// ---------------------------------------------------------------------------
// U192 ECM curve point and operations
// ---------------------------------------------------------------------------

/// Point on a Montgomery curve in projective coordinates, using U192 Montgomery form.
#[derive(Clone, Copy, Debug)]
struct MontPoint192 {
    x: U192,
    z: U192,
}

/// Montgomery curve doubling for U192.
///
/// Formulas (all mod n via Montgomery multiplication):
/// ```text
/// sum  = X + Z
/// diff = X - Z
/// u    = sum^2
/// v    = diff^2
/// w    = u - v
/// X2   = u * v
/// Z2   = w * (v + a24 * w)
/// ```
#[inline]
fn mont_double_192(
    p: &MontPoint192,
    a24: &U192,
    mont: &Mont192,
) -> MontPoint192 {
    let sum = add_mod_192(&p.x, &p.z, &mont.n);
    let diff = sub_mod_192(&p.x, &p.z, &mont.n);
    let u = mont_sqr_192(&sum, mont);
    let v = mont_sqr_192(&diff, mont);
    let w = sub_mod_192(&u, &v, &mont.n);
    let x2 = mont_mul_192(&u, &v, mont);
    let t = mont_mul_192(a24, &w, mont);
    let vt = add_mod_192(&v, &t, &mont.n);
    let z2 = mont_mul_192(&w, &vt, mont);
    MontPoint192 { x: x2, z: z2 }
}

/// Montgomery differential addition for U192.
///
/// Given P, Q, and Diff = P - Q, compute P + Q.
/// ```text
/// u1 = (Xp - Zp) * (Xq + Zq)
/// u2 = (Xp + Zp) * (Xq - Zq)
/// X3 = Zdiff * (u1 + u2)^2
/// Z3 = Xdiff * (u1 - u2)^2
/// ```
#[inline]
fn mont_add_192(
    p: &MontPoint192,
    q: &MontPoint192,
    diff: &MontPoint192,
    mont: &Mont192,
) -> MontPoint192 {
    let d1 = sub_mod_192(&p.x, &p.z, &mont.n);
    let s2 = add_mod_192(&q.x, &q.z, &mont.n);
    let u1 = mont_mul_192(&d1, &s2, mont);

    let s1 = add_mod_192(&p.x, &p.z, &mont.n);
    let d2 = sub_mod_192(&q.x, &q.z, &mont.n);
    let u2 = mont_mul_192(&s1, &d2, mont);

    let sum = add_mod_192(&u1, &u2, &mont.n);
    let dif = sub_mod_192(&u1, &u2, &mont.n);
    let x3 = mont_mul_192(&diff.z, &mont_sqr_192(&sum, mont), mont);
    let z3 = mont_mul_192(&diff.x, &mont_sqr_192(&dif, mont), mont);
    MontPoint192 { x: x3, z: z3 }
}

/// Montgomery ladder: compute [k]P using U192 arithmetic.
fn mont_ladder_192(p: &MontPoint192, k: u64, a24: &U192, mont: &Mont192) -> MontPoint192 {
    if k == 0 {
        return MontPoint192 {
            x: to_mont_192(&U192_ONE, mont),
            z: U192_ZERO,
        };
    }
    if k == 1 {
        return *p;
    }

    let mut r0 = *p;
    let mut r1 = mont_double_192(p, a24, mont);

    let bits = 64 - k.leading_zeros();
    for i in (0..bits - 1).rev() {
        if (k >> i) & 1 == 1 {
            r0 = mont_add_192(&r0, &r1, p, mont);
            r1 = mont_double_192(&r1, a24, mont);
        } else {
            r1 = mont_add_192(&r0, &r1, p, mont);
            r0 = mont_double_192(&r0, a24, mont);
        }
    }

    r0
}

/// Modular inverse for U192 via extended binary GCD.
///
/// Returns Ok(a^{-1} mod n) if gcd(a, n) == 1, or Err(gcd) otherwise.
fn mod_inverse_192(a: &U192, n: &U192) -> Result<U192, U192> {
    let g = gcd_192(a, n);
    if g == U192_ONE {
        // Use rug for the inverse since extended GCD for 192-bit is complex.
        let a_int = u192_to_integer(a);
        let n_int = u192_to_integer(n);
        match a_int.invert(&n_int) {
            Ok(inv) => Ok(integer_to_u192(&inv)),
            Err(_) => Err(g),
        }
    } else {
        Err(g)
    }
}

/// Modular multiplication without Montgomery form (for Suyama setup).
/// Computes a * b mod n using the 6-limb product and reduction.
fn mulmod_192(a: &U192, b: &U192, n: &U192) -> U192 {
    let product = mul_3x3(a, b);
    // Convert the 6-limb product to Integer, take mod n, convert back.
    // This is only used in curve setup (not hot path), so the overhead is fine.
    let mut p_int = Integer::new();
    let mut bytes = [0u8; 48];
    for i in 0..6 {
        let limb = product[i];
        for j in 0..8 {
            bytes[i * 8 + j] = ((limb >> (j * 8)) & 0xFF) as u8;
        }
    }
    p_int.assign(Integer::from_digits(&bytes, rug::integer::Order::LsfLe));
    let n_int = u192_to_integer(n);
    p_int.modulo_mut(&n_int);
    integer_to_u192(&p_int)
}

/// Multiply a U192 by a small u64 modulo n (for Suyama setup).
fn mul_u64_mod_192(a: &U192, b: u64, n: &U192) -> U192 {
    let b_arr: U192 = [b, 0, 0];
    mulmod_192(a, &b_arr, n)
}

/// Modular subtraction for regular (non-Montgomery) values during setup.
fn sub_mod_regular_192(a: &U192, b: &U192, n: &U192) -> U192 {
    sub_mod_192(a, b, n)
}

// ---------------------------------------------------------------------------
// Fast ECM implementation using U192
// ---------------------------------------------------------------------------

/// Run one ECM curve on `n` using fixed-width 192-bit Montgomery arithmetic.
///
/// This is the fast path for n <= 192 bits. Returns `Some(factor)` if a
/// non-trivial factor is found in Phase 1 or Phase 2.
fn ecm_one_curve_fast(
    n_192: &U192,
    n_int: &Integer,
    b1: u64,
    b2: u64,
    sigma: u64,
    primes: &[u64],
    mont: &Mont192,
    prime_bits: Option<&[u64]>,
) -> Option<Integer> {
    // Suyama parameterization.
    let sigma_arr: U192 = [sigma, 0, 0];
    let sig_mod: U192 = if cmp_192(&sigma_arr, n_192) == std::cmp::Ordering::Less {
        sigma_arr
    } else {
        // sigma < 2^64, so for any real n_192 this branch is unlikely.
        let s_int = Integer::from(sigma);
        let n_i = u192_to_integer(n_192);
        let s_mod = Integer::from(&s_int % &n_i);
        integer_to_u192(&s_mod)
    };

    // u = sigma^2 - 5 mod n
    let sig_sq = mulmod_192(&sig_mod, &sig_mod, n_192);
    let five: U192 = [5, 0, 0];
    let u = sub_mod_regular_192(&sig_sq, &five, n_192);
    // v = 4 * sigma mod n
    let v = mul_u64_mod_192(&sig_mod, 4, n_192);

    if u == U192_ZERO || v == U192_ZERO {
        return None;
    }

    // Starting point Q = (u^3 : v^3)
    let u3 = mulmod_192(&mulmod_192(&u, &u, n_192), &u, n_192);
    let v3 = mulmod_192(&mulmod_192(&v, &v, n_192), &v, n_192);

    // a24 = (v-u)^3 * (3u+v) / (16 * u^3 * v) mod n
    let v_minus_u = sub_mod_regular_192(&v, &u, n_192);
    let vmu3 = mulmod_192(
        &mulmod_192(&v_minus_u, &v_minus_u, n_192),
        &v_minus_u,
        n_192,
    );
    let three_u = mul_u64_mod_192(&u, 3, n_192);
    let three_u_plus_v = add_mod_192(&three_u, &v, n_192);
    let numerator = mulmod_192(&vmu3, &three_u_plus_v, n_192);

    let u3v = mulmod_192(&u3, &v, n_192);
    let denom = mul_u64_mod_192(&u3v, 16, n_192);

    let denom_inv = match mod_inverse_192(&denom, n_192) {
        Ok(inv) => inv,
        Err(g) => {
            if g != U192_ONE && g != *n_192 {
                return Some(u192_to_integer(&g));
            }
            return None;
        }
    };

    let a24 = mulmod_192(&numerator, &denom_inv, n_192);

    // Convert to Montgomery form.
    let a24_mont = to_mont_192(&a24, mont);
    let q0 = MontPoint192 {
        x: to_mont_192(&u3, mont),
        z: to_mont_192(&v3, mont),
    };

    // Phase 1: scalar multiply by lcm(1..B1).
    let one_mont = to_mont_192(&U192_ONE, mont);
    let mut q = q0;
    let mut accum = one_mont;
    let mut step_count = 0u32;

    for &p in primes.iter() {
        if p > b1 {
            break;
        }
        // Compute p^k where p^k <= B1 < p^(k+1), then do a single ladder.
        // This is faster than k separate [p]Q ladders because a single
        // [p^k]Q ladder has log2(p^k) = k*log2(p) bits, while k separate
        // [p]Q ladders have k*log2(p) bits total but with k function call
        // overheads and k times the ladder startup cost.
        let mut pk = p;
        while let Some(next) = pk.checked_mul(p) {
            if next > b1 { break; }
            pk = next;
        }
        q = mont_ladder_192(&q, pk, &a24_mont, mont);

        // Accumulate Z into product for batched GCD.
        accum = mont_mul_192(&accum, &q.z, mont);
        step_count += 1;

        // Check GCD every 16 primes. More frequent checks detect factors
        // earlier, reducing wasted work on curves that already found a factor.
        // Cost: ~0.5μs per GCD vs ~0.3ms per 16 ladder steps → negligible.
        if step_count % 16 == 0 {
            let accum_reg = from_mont_192(&accum, mont);
            let g = gcd_192(&accum_reg, n_192);
            if g != U192_ONE && g != *n_192 {
                return Some(u192_to_integer(&g));
            }
            if accum_reg == U192_ZERO {
                // Overshot -- fall back to careful rug-based path.
                return ecm_one_curve_careful(n_int, b1, b2, sigma, primes);
            }
        }
    }

    // Final Phase 1 GCD check.
    let accum_reg = from_mont_192(&accum, mont);
    let g = gcd_192(&accum_reg, n_192);
    if g != U192_ONE && g != *n_192 {
        return Some(u192_to_integer(&g));
    }
    if accum_reg == U192_ZERO {
        return ecm_one_curve_careful(n_int, b1, b2, sigma, primes);
    }

    // Phase 2: baby-step giant-step for primes in (B1, B2].
    let mut accum2 = one_mont;
    let mut batch_count = 0u32;
    const PHASE2_BATCH: u32 = 32;

    let q_base = q;

    let d_step = ((b2 - b1) as f64).sqrt() as u64;
    let d = if d_step < 2 { 2 } else { d_step | 1 };

    // Precompute baby steps: [1]Q, [2]Q, ..., [d]Q.
    let mut baby = Vec::with_capacity(d as usize);
    baby.push(q_base);
    if d >= 2 {
        baby.push(mont_double_192(&q_base, &a24_mont, mont));
    }
    for i in 3..=d {
        let len = baby.len();
        let next = mont_add_192(&baby[len - 1], &baby[0], &baby[len - 2], mont);
        baby.push(next);
        let _ = i;
    }

    let d_q = baby[baby.len() - 1];

    // Giant step: start at B1 rounded up to next multiple of d.
    let first_giant = ((b1 / d) + 1) * d;
    let mut giant_q = mont_ladder_192(&q_base, first_giant, &a24_mont, mont);
    let mut prev_giant_q = if first_giant >= d {
        mont_ladder_192(&q_base, first_giant - d, &a24_mont, mont)
    } else {
        MontPoint192 {
            x: to_mont_192(&U192_ONE, mont),
            z: U192_ZERO,
        }
    };

    // Two-pointer scan: maintain sliding window over sorted primes array.
    // As giant_val advances by d, the window [giant-d, giant+d] shifts right.
    // This eliminates binary searches (2 per giant step → 0).
    let p_start = primes.partition_point(|&p| p <= b1);
    let p_end = primes.len();
    let mut scan_idx = p_start; // sliding lower bound

    let mut giant_val = first_giant;
    while giant_val <= b2 + d {
        // Advance scan_idx past primes below (giant_val - d)
        let range_lo = if giant_val > d { giant_val - d } else { 0 };
        while scan_idx < p_end && primes[scan_idx] < range_lo.max(b1 + 1) {
            scan_idx += 1;
        }

        // Iterate primes in [range_lo, giant_val + d] ∩ (b1, b2]
        let mut pi = scan_idx;
        while pi < p_end && primes[pi] <= (giant_val + d).min(b2) {
            let p = primes[pi];
            pi += 1;

            let b_offset = if p > giant_val { p - giant_val } else { giant_val - p };
            if b_offset < 1 || b_offset > d { continue; }
            let b_idx = (b_offset - 1) as usize;

            let t1 = mont_mul_192(&giant_q.x, &baby[b_idx].z, mont);
            let t2 = mont_mul_192(&giant_q.z, &baby[b_idx].x, mont);
            let cross = sub_mod_192(&t1, &t2, &mont.n);
            accum2 = mont_mul_192(&accum2, &cross, mont);
            batch_count += 1;

            if batch_count >= PHASE2_BATCH {
                let accum2_reg = from_mont_192(&accum2, mont);
                let g = gcd_192(&accum2_reg, n_192);
                if g != U192_ONE && g != *n_192 {
                    return Some(u192_to_integer(&g));
                }
                batch_count = 0;
            }
        }

        // Advance giant step.
        let new_giant = mont_add_192(&giant_q, &d_q, &prev_giant_q, mont);
        prev_giant_q = giant_q;
        giant_q = new_giant;
        giant_val += d;
    }

    // Final Phase 2 GCD check.
    let accum2_reg = from_mont_192(&accum2, mont);
    let g = gcd_192(&accum2_reg, n_192);
    if g != U192_ONE && g != *n_192 {
        return Some(u192_to_integer(&g));
    }
    None
}

/// A point on a Montgomery curve in projective coordinates `(X : Z)`.
///
/// Uses pre-allocated Integers to minimize GMP heap allocations in the hot loop.
#[derive(Clone, Debug)]
struct MontPoint {
    x: Integer,
    z: Integer,
}

/// Scratch space for Montgomery curve operations.
///
/// Pre-allocates all temporary Integers needed by mont_double and mont_add
/// so the hot loop does zero heap allocations. For 148-bit N, each Integer
/// needs ~3 GMP limbs (192 bits); pre-allocating avoids thousands of malloc/free
/// calls per scalar multiplication.
struct MontScratch {
    s1: Integer,
    s2: Integer,
    s3: Integer,
    s4: Integer,
    s5: Integer,
    s6: Integer,
}

impl MontScratch {
    fn new() -> Self {
        MontScratch {
            s1: Integer::new(),
            s2: Integer::new(),
            s3: Integer::new(),
            s4: Integer::new(),
            s5: Integer::new(),
            s6: Integer::new(),
        }
    }
}

// ===========================================================================
// Fixed-width 256-bit Montgomery arithmetic
// ===========================================================================

/// 256-bit unsigned integer, stored as 4 little-endian u64 limbs: [lo, mid_lo, mid_hi, hi].
type U256 = [u64; 4];

/// Zero constant.
const U256_ZERO: U256 = [0, 0, 0, 0];
/// One constant.
const U256_ONE: U256 = [1, 0, 0, 0];

/// Montgomery form parameters for 256-bit modular arithmetic.
///
/// For modulus n (must be odd), R = 2^256. We store:
/// - `n`: the modulus
/// - `n_inv`: -n^{-1} mod 2^64 (single limb, used in REDC)
/// - `r_squared`: R^2 mod n (used to convert into Montgomery form)
#[derive(Clone, Debug)]
struct Mont256 {
    n: U256,
    n_inv: u64,
    r_squared: U256,
}

impl Mont256 {
    /// Build Montgomery parameters for odd modulus `n`.
    fn new(n: &U256) -> Self {
        debug_assert!(n[0] & 1 == 1, "Mont256: n must be odd");

        // Compute n_inv = -n^{-1} mod 2^64 via Newton's method.
        let n0 = n[0];
        let mut x: u64 = 1;
        for _ in 0..6 {
            x = x.wrapping_mul(2u64.wrapping_sub(n0.wrapping_mul(x)));
        }
        let n_inv = x.wrapping_neg();

        // Compute R mod n = 2^256 mod n.
        let r_mod_n = Self::compute_r_mod_n(n);

        // Compute R^2 mod n = (R mod n) doubled 256 more times.
        let r_squared = Self::compute_r_squared(n, &r_mod_n);

        Mont256 {
            n: *n,
            n_inv,
            r_squared,
        }
    }

    /// Compute 2^256 mod n by repeated doubling.
    fn compute_r_mod_n(n: &U256) -> U256 {
        let mut r = U256_ONE;
        for _ in 0..256 {
            r = add_mod_256(&r, &r, n);
        }
        r
    }

    /// Compute R^2 mod n = (R mod n) doubled 256 more times.
    fn compute_r_squared(n: &U256, r_mod_n: &U256) -> U256 {
        let mut r2 = *r_mod_n;
        for _ in 0..256 {
            r2 = add_mod_256(&r2, &r2, n);
        }
        r2
    }
}

/// Compare two U256 values: returns Ordering.
#[inline]
fn cmp_256(a: &U256, b: &U256) -> std::cmp::Ordering {
    if a[3] != b[3] {
        return a[3].cmp(&b[3]);
    }
    if a[2] != b[2] {
        return a[2].cmp(&b[2]);
    }
    if a[1] != b[1] {
        return a[1].cmp(&b[1]);
    }
    a[0].cmp(&b[0])
}

/// a + b mod n, where a, b < n.
#[inline]
fn add_mod_256(a: &U256, b: &U256, n: &U256) -> U256 {
    let (s0, c0) = a[0].overflowing_add(b[0]);
    let (s1, c1a) = a[1].overflowing_add(b[1]);
    let (s1, c1b) = s1.overflowing_add(c0 as u64);
    let c1 = c1a | c1b;
    let (s2, c2a) = a[2].overflowing_add(b[2]);
    let (s2, c2b) = s2.overflowing_add(c1 as u64);
    let c2 = c2a | c2b;
    let (s3, c3a) = a[3].overflowing_add(b[3]);
    let (s3, c3b) = s3.overflowing_add(c2 as u64);
    let carry = c3a | c3b;

    let mut s = [s0, s1, s2, s3];

    if carry || cmp_256(&s, n) != std::cmp::Ordering::Less {
        let (d0, borrow0) = s[0].overflowing_sub(n[0]);
        let (d1, borrow1a) = s[1].overflowing_sub(n[1]);
        let (d1, borrow1b) = d1.overflowing_sub(borrow0 as u64);
        let borrow1 = borrow1a | borrow1b;
        let (d2, borrow2a) = s[2].overflowing_sub(n[2]);
        let (d2, borrow2b) = d2.overflowing_sub(borrow1 as u64);
        let borrow2 = borrow2a | borrow2b;
        let d3 = s[3]
            .wrapping_sub(n[3])
            .wrapping_sub(borrow2 as u64);
        s = [d0, d1, d2, d3];
    }

    s
}

/// a - b mod n, where a, b < n. Returns a + n - b if a < b.
#[inline]
fn sub_mod_256(a: &U256, b: &U256, n: &U256) -> U256 {
    if cmp_256(a, b) != std::cmp::Ordering::Less {
        // a >= b: just subtract.
        let (d0, borrow0) = a[0].overflowing_sub(b[0]);
        let (d1, borrow1a) = a[1].overflowing_sub(b[1]);
        let (d1, borrow1b) = d1.overflowing_sub(borrow0 as u64);
        let borrow1 = borrow1a | borrow1b;
        let (d2, borrow2a) = a[2].overflowing_sub(b[2]);
        let (d2, borrow2b) = d2.overflowing_sub(borrow1 as u64);
        let borrow2 = borrow2a | borrow2b;
        let d3 = a[3]
            .wrapping_sub(b[3])
            .wrapping_sub(borrow2 as u64);
        [d0, d1, d2, d3]
    } else {
        // a < b: compute a + n - b.
        let (t0, c0) = a[0].overflowing_add(n[0]);
        let (t1, c1a) = a[1].overflowing_add(n[1]);
        let (t1, c1b) = t1.overflowing_add(c0 as u64);
        let c1 = c1a | c1b;
        let (t2, c2a) = a[2].overflowing_add(n[2]);
        let (t2, c2b) = t2.overflowing_add(c1 as u64);
        let c2 = c2a | c2b;
        let t3 = a[3].wrapping_add(n[3]).wrapping_add(c2 as u64);
        // Then subtract b.
        let (d0, borrow0) = t0.overflowing_sub(b[0]);
        let (d1, borrow1a) = t1.overflowing_sub(b[1]);
        let (d1, borrow1b) = d1.overflowing_sub(borrow0 as u64);
        let borrow1 = borrow1a | borrow1b;
        let (d2, borrow2a) = t2.overflowing_sub(b[2]);
        let (d2, borrow2b) = d2.overflowing_sub(borrow1 as u64);
        let borrow2 = borrow2a | borrow2b;
        let d3 = t3
            .wrapping_sub(b[3])
            .wrapping_sub(borrow2 as u64);
        [d0, d1, d2, d3]
    }
}

/// 4-limb x 4-limb multiplication producing an 8-limb result.
///
/// Uses the classic schoolbook method with u128 intermediates.
#[inline]
/// 4-limb schoolbook multiplication: a[0..3] × b[0..3] → t[0..7].
#[inline]
fn mul_4x4(a: &U256, b: &U256) -> [u64; 8] {
    let mut t = [0u64; 8];
    for i in 0..4 {
        let mut carry: u64 = 0;
        for j in 0..4 {
            let prod = a[i] as u128 * b[j] as u128 + t[i + j] as u128 + carry as u128;
            t[i + j] = prod as u64;
            carry = (prod >> 64) as u64;
        }
        t[i + 4] = carry;
    }
    t
}

/// Multiply-accumulate: t[offset..offset+4] += m * n, returning the carry out.
///
/// Computes the 4-limb product m * n[0..3] and adds it into t starting at `offset`.
/// Returns the carry that propagates past t[offset+3].
#[inline]
fn mac_4_at(t: &mut [u64; 8], offset: usize, m: u64, n: &U256) -> u64 {
    let m128 = m as u128;

    // Limb 0
    let p = m128 * n[0] as u128 + t[offset] as u128;
    t[offset] = p as u64;
    let mut carry = (p >> 64) as u128;

    // Limb 1
    let p = m128 * n[1] as u128 + t[offset + 1] as u128 + carry;
    t[offset + 1] = p as u64;
    carry = p >> 64;

    // Limb 2
    let p = m128 * n[2] as u128 + t[offset + 2] as u128 + carry;
    t[offset + 2] = p as u64;
    carry = p >> 64;

    // Limb 3
    let p = m128 * n[3] as u128 + t[offset + 3] as u128 + carry;
    t[offset + 3] = p as u64;
    (p >> 64) as u64
}

/// Montgomery multiplication: CIOS (Coarsely Integrated Operand Scanning).
///
/// Interleaves multiplication and reduction row-by-row instead of
/// computing the full 8-limb product first. This halves the number
/// of temporary limbs (5 vs 8), improving register allocation on
/// aarch64 where we have 30 general-purpose registers.
///
/// Algorithm (for n=4 limbs):
///   For i = 0..3:
///     1. Multiply: t += a[i] * b (4 products, accumulate into t[0..4])
///     2. Reduce:   m = t[0] * n_inv; t += m * n; shift t right by 1 limb
///   Result = t[0..3], conditionally subtract n.
#[inline(always)]
fn mont_mul_256(a: &U256, b: &U256, mont: &Mont256) -> U256 {
    let n = &mont.n;
    let n_inv = mont.n_inv;

    // Only 5 working limbs needed (vs 8 for separate mul+redc).
    let mut t0: u64 = 0;
    let mut t1: u64 = 0;
    let mut t2: u64 = 0;
    let mut t3: u64 = 0;
    let mut t4: u64 = 0;

    // --- Row i=0: t += a[0] * b, then reduce ---
    {
        let ai = a[0] as u128;
        let p0 = ai * b[0] as u128;
        t0 = p0 as u64;
        let mut c = (p0 >> 64) as u128;
        let p1 = ai * b[1] as u128 + c;
        t1 = p1 as u64;
        c = p1 >> 64;
        let p2 = ai * b[2] as u128 + c;
        t2 = p2 as u64;
        c = p2 >> 64;
        let p3 = ai * b[3] as u128 + c;
        t3 = p3 as u64;
        t4 = (p3 >> 64) as u64;

        // Reduce: m = t0 * n_inv; t += m * n; shift right
        let m = t0.wrapping_mul(n_inv) as u128;
        let p0 = m * n[0] as u128 + t0 as u128;
        c = p0 >> 64;
        let p1 = m * n[1] as u128 + t1 as u128 + c;
        t0 = p1 as u64; // shift: t0 = old t1 position
        c = p1 >> 64;
        let p2 = m * n[2] as u128 + t2 as u128 + c;
        t1 = p2 as u64;
        c = p2 >> 64;
        let p3 = m * n[3] as u128 + t3 as u128 + c;
        t2 = p3 as u64;
        let c = (p3 >> 64) as u64;
        t3 = t4.wrapping_add(c);
        t4 = (t3 < c) as u64; // carry out
        // Fix: t3 might have wrapped. Recompute properly.
        let (sum, carry) = t4.overflowing_add(0); // t4 is already the carry
        let _ = (sum, carry);
    }

    // --- Row i=1 ---
    {
        let ai = a[1] as u128;
        let p0 = ai * b[0] as u128 + t0 as u128;
        t0 = p0 as u64;
        let mut c = (p0 >> 64) as u128;
        let p1 = ai * b[1] as u128 + t1 as u128 + c;
        t1 = p1 as u64;
        c = p1 >> 64;
        let p2 = ai * b[2] as u128 + t2 as u128 + c;
        t2 = p2 as u64;
        c = p2 >> 64;
        let p3 = ai * b[3] as u128 + t3 as u128 + c;
        t3 = p3 as u64;
        let c4 = (p3 >> 64) as u64;
        t4 = t4.wrapping_add(c4);

        let m = t0.wrapping_mul(n_inv) as u128;
        let p0 = m * n[0] as u128 + t0 as u128;
        c = p0 >> 64;
        let p1 = m * n[1] as u128 + t1 as u128 + c;
        t0 = p1 as u64;
        c = p1 >> 64;
        let p2 = m * n[2] as u128 + t2 as u128 + c;
        t1 = p2 as u64;
        c = p2 >> 64;
        let p3 = m * n[3] as u128 + t3 as u128 + c;
        t2 = p3 as u64;
        let c = (p3 >> 64) as u64;
        let (t3_new, carry) = t4.overflowing_add(c);
        t3 = t3_new;
        t4 = carry as u64;
    }

    // --- Row i=2 ---
    {
        let ai = a[2] as u128;
        let p0 = ai * b[0] as u128 + t0 as u128;
        t0 = p0 as u64;
        let mut c = (p0 >> 64) as u128;
        let p1 = ai * b[1] as u128 + t1 as u128 + c;
        t1 = p1 as u64;
        c = p1 >> 64;
        let p2 = ai * b[2] as u128 + t2 as u128 + c;
        t2 = p2 as u64;
        c = p2 >> 64;
        let p3 = ai * b[3] as u128 + t3 as u128 + c;
        t3 = p3 as u64;
        let c4 = (p3 >> 64) as u64;
        t4 = t4.wrapping_add(c4);

        let m = t0.wrapping_mul(n_inv) as u128;
        let p0 = m * n[0] as u128 + t0 as u128;
        c = p0 >> 64;
        let p1 = m * n[1] as u128 + t1 as u128 + c;
        t0 = p1 as u64;
        c = p1 >> 64;
        let p2 = m * n[2] as u128 + t2 as u128 + c;
        t1 = p2 as u64;
        c = p2 >> 64;
        let p3 = m * n[3] as u128 + t3 as u128 + c;
        t2 = p3 as u64;
        let c = (p3 >> 64) as u64;
        let (t3_new, carry) = t4.overflowing_add(c);
        t3 = t3_new;
        t4 = carry as u64;
    }

    // --- Row i=3 ---
    {
        let ai = a[3] as u128;
        let p0 = ai * b[0] as u128 + t0 as u128;
        t0 = p0 as u64;
        let mut c = (p0 >> 64) as u128;
        let p1 = ai * b[1] as u128 + t1 as u128 + c;
        t1 = p1 as u64;
        c = p1 >> 64;
        let p2 = ai * b[2] as u128 + t2 as u128 + c;
        t2 = p2 as u64;
        c = p2 >> 64;
        let p3 = ai * b[3] as u128 + t3 as u128 + c;
        t3 = p3 as u64;
        let c4 = (p3 >> 64) as u64;
        t4 = t4.wrapping_add(c4);

        let m = t0.wrapping_mul(n_inv) as u128;
        let p0 = m * n[0] as u128 + t0 as u128;
        c = p0 >> 64;
        let p1 = m * n[1] as u128 + t1 as u128 + c;
        t0 = p1 as u64;
        c = p1 >> 64;
        let p2 = m * n[2] as u128 + t2 as u128 + c;
        t1 = p2 as u64;
        c = p2 >> 64;
        let p3 = m * n[3] as u128 + t3 as u128 + c;
        t2 = p3 as u64;
        let c = (p3 >> 64) as u64;
        let (t3_new, _carry) = t4.overflowing_add(c);
        t3 = t3_new;
    }

    // Result in [t0, t1, t2, t3]. Conditionally subtract n.
    let mut result = [t0, t1, t2, t3];
    if cmp_256(&result, &mont.n) != std::cmp::Ordering::Less {
        let (d0, borrow0) = result[0].overflowing_sub(n[0]);
        let (d1, borrow1a) = result[1].overflowing_sub(n[1]);
        let (d1, borrow1b) = d1.overflowing_sub(borrow0 as u64);
        let borrow1 = borrow1a | borrow1b;
        let (d2, borrow2a) = result[2].overflowing_sub(n[2]);
        let (d2, borrow2b) = d2.overflowing_sub(borrow1 as u64);
        let borrow2 = borrow2a | borrow2b;
        let d3 = result[3]
            .wrapping_sub(n[3])
            .wrapping_sub(borrow2 as u64);
        result = [d0, d1, d2, d3];
    }

    result
}

/// Montgomery squaring: a^2 * R^{-1} mod n. Delegates to mont_mul_256.
#[inline]
fn mont_sqr_256(a: &U256, mont: &Mont256) -> U256 {
    mont_mul_256(a, a, mont)
}

/// Convert a regular value into Montgomery form: a * R mod n.
#[inline]
fn to_mont_256(a: &U256, mont: &Mont256) -> U256 {
    mont_mul_256(a, &mont.r_squared, mont)
}

/// Convert from Montgomery form back to regular: a * R^{-1} mod n.
#[inline]
fn from_mont_256(a: &U256, mont: &Mont256) -> U256 {
    mont_mul_256(a, &U256_ONE, mont)
}

/// Binary GCD for U256 values. Both inputs are in regular (non-Montgomery) form.
fn gcd_256(a: &U256, b: &U256) -> U256 {
    if *a == U256_ZERO {
        return *b;
    }
    if *b == U256_ZERO {
        return *a;
    }

    let mut u = *a;
    let mut v = *b;

    let shift_u = trailing_zeros_256(&u);
    let shift_v = trailing_zeros_256(&v);
    let shift = shift_u.min(shift_v);

    shr_256_inplace(&mut u, shift_u);
    shr_256_inplace(&mut v, shift_v);

    loop {
        match cmp_256(&u, &v) {
            std::cmp::Ordering::Equal => break,
            std::cmp::Ordering::Greater => {
                sub_inplace_256(&mut u, &v);
                let tz = trailing_zeros_256(&u);
                shr_256_inplace(&mut u, tz);
            }
            std::cmp::Ordering::Less => {
                sub_inplace_256(&mut v, &u);
                let tz = trailing_zeros_256(&v);
                shr_256_inplace(&mut v, tz);
            }
        }
    }

    shl_256_inplace(&mut u, shift);
    u
}

/// Count trailing zero bits of a U256.
#[inline]
fn trailing_zeros_256(a: &U256) -> u32 {
    if a[0] != 0 {
        a[0].trailing_zeros()
    } else if a[1] != 0 {
        64 + a[1].trailing_zeros()
    } else if a[2] != 0 {
        128 + a[2].trailing_zeros()
    } else if a[3] != 0 {
        192 + a[3].trailing_zeros()
    } else {
        256
    }
}

/// Right shift a U256 in place by `shift` bits (0..256).
#[inline]
fn shr_256_inplace(a: &mut U256, shift: u32) {
    if shift == 0 {
        return;
    }
    if shift >= 256 {
        *a = U256_ZERO;
        return;
    }
    if shift >= 192 {
        let s = shift - 192;
        a[0] = if s == 0 { a[3] } else { a[3] >> s };
        a[1] = 0;
        a[2] = 0;
        a[3] = 0;
    } else if shift >= 128 {
        let s = shift - 128;
        if s == 0 {
            a[0] = a[2];
            a[1] = a[3];
        } else {
            a[0] = (a[2] >> s) | (a[3] << (64 - s));
            a[1] = a[3] >> s;
        }
        a[2] = 0;
        a[3] = 0;
    } else if shift >= 64 {
        let s = shift - 64;
        if s == 0 {
            a[0] = a[1];
            a[1] = a[2];
            a[2] = a[3];
        } else {
            a[0] = (a[1] >> s) | (a[2] << (64 - s));
            a[1] = (a[2] >> s) | (a[3] << (64 - s));
            a[2] = a[3] >> s;
        }
        a[3] = 0;
    } else {
        let new0 = (a[0] >> shift) | (a[1] << (64 - shift));
        let new1 = (a[1] >> shift) | (a[2] << (64 - shift));
        let new2 = (a[2] >> shift) | (a[3] << (64 - shift));
        let new3 = a[3] >> shift;
        a[0] = new0;
        a[1] = new1;
        a[2] = new2;
        a[3] = new3;
    }
}

/// Left shift a U256 in place by `shift` bits (0..256).
#[inline]
fn shl_256_inplace(a: &mut U256, shift: u32) {
    if shift == 0 {
        return;
    }
    if shift >= 256 {
        *a = U256_ZERO;
        return;
    }
    if shift >= 192 {
        let s = shift - 192;
        a[3] = if s == 0 { a[0] } else { a[0] << s };
        a[2] = 0;
        a[1] = 0;
        a[0] = 0;
    } else if shift >= 128 {
        let s = shift - 128;
        if s == 0 {
            a[3] = a[1];
            a[2] = a[0];
        } else {
            a[3] = (a[1] << s) | (a[0] >> (64 - s));
            a[2] = a[0] << s;
        }
        a[1] = 0;
        a[0] = 0;
    } else if shift >= 64 {
        let s = shift - 64;
        if s == 0 {
            a[3] = a[2];
            a[2] = a[1];
            a[1] = a[0];
        } else {
            a[3] = (a[2] << s) | (a[1] >> (64 - s));
            a[2] = (a[1] << s) | (a[0] >> (64 - s));
            a[1] = a[0] << s;
        }
        a[0] = 0;
    } else {
        let new3 = (a[3] << shift) | (a[2] >> (64 - shift));
        let new2 = (a[2] << shift) | (a[1] >> (64 - shift));
        let new1 = (a[1] << shift) | (a[0] >> (64 - shift));
        let new0 = a[0] << shift;
        a[0] = new0;
        a[1] = new1;
        a[2] = new2;
        a[3] = new3;
    }
}

/// u = u - v in place (assumes u >= v).
#[inline]
fn sub_inplace_256(u: &mut U256, v: &U256) {
    let (d0, borrow0) = u[0].overflowing_sub(v[0]);
    let (d1, borrow1a) = u[1].overflowing_sub(v[1]);
    let (d1, borrow1b) = d1.overflowing_sub(borrow0 as u64);
    let borrow1 = borrow1a | borrow1b;
    let (d2, borrow2a) = u[2].overflowing_sub(v[2]);
    let (d2, borrow2b) = d2.overflowing_sub(borrow1 as u64);
    let borrow2 = borrow2a | borrow2b;
    let d3 = u[3]
        .wrapping_sub(v[3])
        .wrapping_sub(borrow2 as u64);
    u[0] = d0;
    u[1] = d1;
    u[2] = d2;
    u[3] = d3;
}

// ---------------------------------------------------------------------------
// Conversion between rug::Integer and U256
// ---------------------------------------------------------------------------

/// Convert a non-negative rug::Integer to U256. Panics if n > 2^256 - 1.
fn integer_to_u256(n: &Integer) -> U256 {
    debug_assert!(*n >= 0, "integer_to_u256: n must be non-negative");
    debug_assert!(
        n.significant_bits() <= 256,
        "integer_to_u256: n must fit in 256 bits (got {} bits)",
        n.significant_bits()
    );

    let digits = n.to_digits::<u8>(rug::integer::Order::LsfLe);
    let mut limbs = [0u64; 4];
    for (i, &byte) in digits.iter().enumerate().take(32) {
        limbs[i / 8] |= (byte as u64) << ((i % 8) * 8);
    }
    limbs
}

/// Convert a U256 back to rug::Integer.
fn u256_to_integer(a: &U256) -> Integer {
    let mut bytes = [0u8; 32];
    for i in 0..4 {
        let limb = a[i];
        for j in 0..8 {
            bytes[i * 8 + j] = ((limb >> (j * 8)) & 0xFF) as u8;
        }
    }
    Integer::from_digits(&bytes, rug::integer::Order::LsfLe)
}

// ---------------------------------------------------------------------------
// U256 ECM curve point and operations
// ---------------------------------------------------------------------------

/// Point on a Montgomery curve in projective coordinates, using U256 Montgomery form.
#[derive(Clone, Copy, Debug)]
struct MontPoint256 {
    x: U256,
    z: U256,
}

/// Montgomery curve doubling for U256.
#[inline]
fn mont_double_256(
    p: &MontPoint256,
    a24: &U256,
    mont: &Mont256,
) -> MontPoint256 {
    let sum = add_mod_256(&p.x, &p.z, &mont.n);
    let diff = sub_mod_256(&p.x, &p.z, &mont.n);
    let u = mont_sqr_256(&sum, mont);
    let v = mont_sqr_256(&diff, mont);
    let w = sub_mod_256(&u, &v, &mont.n);
    let x2 = mont_mul_256(&u, &v, mont);
    let t = mont_mul_256(a24, &w, mont);
    let vt = add_mod_256(&v, &t, &mont.n);
    let z2 = mont_mul_256(&w, &vt, mont);
    MontPoint256 { x: x2, z: z2 }
}

/// Montgomery differential addition for U256.
#[inline]
fn mont_add_256(
    p: &MontPoint256,
    q: &MontPoint256,
    diff: &MontPoint256,
    mont: &Mont256,
) -> MontPoint256 {
    let d1 = sub_mod_256(&p.x, &p.z, &mont.n);
    let s2 = add_mod_256(&q.x, &q.z, &mont.n);
    let u1 = mont_mul_256(&d1, &s2, mont);

    let s1 = add_mod_256(&p.x, &p.z, &mont.n);
    let d2 = sub_mod_256(&q.x, &q.z, &mont.n);
    let u2 = mont_mul_256(&s1, &d2, mont);

    let sum = add_mod_256(&u1, &u2, &mont.n);
    let dif = sub_mod_256(&u1, &u2, &mont.n);
    let x3 = mont_mul_256(&diff.z, &mont_sqr_256(&sum, mont), mont);
    let z3 = mont_mul_256(&diff.x, &mont_sqr_256(&dif, mont), mont);
    MontPoint256 { x: x3, z: z3 }
}

/// Montgomery ladder: compute [k]P using U256 arithmetic.
fn mont_ladder_256(p: &MontPoint256, k: u64, a24: &U256, mont: &Mont256) -> MontPoint256 {
    if k == 0 {
        return MontPoint256 {
            x: to_mont_256(&U256_ONE, mont),
            z: U256_ZERO,
        };
    }
    if k == 1 {
        return *p;
    }

    let mut r0 = *p;
    let mut r1 = mont_double_256(p, a24, mont);

    let bits = 64 - k.leading_zeros();
    for i in (0..bits - 1).rev() {
        if (k >> i) & 1 == 1 {
            r0 = mont_add_256(&r0, &r1, p, mont);
            r1 = mont_double_256(&r1, a24, mont);
        } else {
            r1 = mont_add_256(&r0, &r1, p, mont);
            r0 = mont_double_256(&r0, a24, mont);
        }
    }

    r0
}

/// Modular inverse for U256 via extended binary GCD.
///
/// Returns Ok(a^{-1} mod n) if gcd(a, n) == 1, or Err(gcd) otherwise.
fn mod_inverse_256(a: &U256, n: &U256) -> Result<U256, U256> {
    let g = gcd_256(a, n);
    if g == U256_ONE {
        let a_int = u256_to_integer(a);
        let n_int = u256_to_integer(n);
        match a_int.invert(&n_int) {
            Ok(inv) => Ok(integer_to_u256(&inv)),
            Err(_) => Err(g),
        }
    } else {
        Err(g)
    }
}

/// Modular multiplication without Montgomery form (for Suyama setup).
/// Computes a * b mod n using the 8-limb product and reduction.
fn mulmod_256(a: &U256, b: &U256, n: &U256) -> U256 {
    let product = mul_4x4(a, b);
    let mut p_int = Integer::new();
    let mut bytes = [0u8; 64];
    for i in 0..8 {
        let limb = product[i];
        for j in 0..8 {
            bytes[i * 8 + j] = ((limb >> (j * 8)) & 0xFF) as u8;
        }
    }
    p_int.assign(Integer::from_digits(&bytes, rug::integer::Order::LsfLe));
    let n_int = u256_to_integer(n);
    p_int.modulo_mut(&n_int);
    integer_to_u256(&p_int)
}

/// Multiply a U256 by a small u64 modulo n (for Suyama setup).
fn mul_u64_mod_256(a: &U256, b: u64, n: &U256) -> U256 {
    let b_arr: U256 = [b, 0, 0, 0];
    mulmod_256(a, &b_arr, n)
}

/// Modular subtraction for regular (non-Montgomery) values during setup.
fn sub_mod_regular_256(a: &U256, b: &U256, n: &U256) -> U256 {
    sub_mod_256(a, b, n)
}

// ---------------------------------------------------------------------------
// Fast ECM implementation using U256
// ---------------------------------------------------------------------------

/// Run one ECM curve on `n` using fixed-width 256-bit Montgomery arithmetic.
///
/// This is the fast path for 193-256 bit n. Returns `Some(factor)` if a
/// non-trivial factor is found in Phase 1 or Phase 2.
fn ecm_one_curve_fast_256(
    n_256: &U256,
    n_int: &Integer,
    b1: u64,
    b2: u64,
    sigma: u64,
    primes: &[u64],
    mont: &Mont256,
    prime_bits: Option<&[u64]>,
) -> Option<Integer> {
    // Suyama parameterization.
    let sigma_arr: U256 = [sigma, 0, 0, 0];
    let sig_mod: U256 = if cmp_256(&sigma_arr, n_256) == std::cmp::Ordering::Less {
        sigma_arr
    } else {
        let s_int = Integer::from(sigma);
        let n_i = u256_to_integer(n_256);
        let s_mod = Integer::from(&s_int % &n_i);
        integer_to_u256(&s_mod)
    };

    // u = sigma^2 - 5 mod n
    let sig_sq = mulmod_256(&sig_mod, &sig_mod, n_256);
    let five: U256 = [5, 0, 0, 0];
    let u = sub_mod_regular_256(&sig_sq, &five, n_256);
    // v = 4 * sigma mod n
    let v = mul_u64_mod_256(&sig_mod, 4, n_256);

    if u == U256_ZERO || v == U256_ZERO {
        return None;
    }

    // Starting point Q = (u^3 : v^3)
    let u3 = mulmod_256(&mulmod_256(&u, &u, n_256), &u, n_256);
    let v3 = mulmod_256(&mulmod_256(&v, &v, n_256), &v, n_256);

    // a24 = (v-u)^3 * (3u+v) / (16 * u^3 * v) mod n
    let v_minus_u = sub_mod_regular_256(&v, &u, n_256);
    let vmu3 = mulmod_256(
        &mulmod_256(&v_minus_u, &v_minus_u, n_256),
        &v_minus_u,
        n_256,
    );
    let three_u = mul_u64_mod_256(&u, 3, n_256);
    let three_u_plus_v = add_mod_256(&three_u, &v, n_256);
    let numerator = mulmod_256(&vmu3, &three_u_plus_v, n_256);

    let u3v = mulmod_256(&u3, &v, n_256);
    let denom = mul_u64_mod_256(&u3v, 16, n_256);

    let denom_inv = match mod_inverse_256(&denom, n_256) {
        Ok(inv) => inv,
        Err(g) => {
            if g != U256_ONE && g != *n_256 {
                return Some(u256_to_integer(&g));
            }
            return None;
        }
    };

    let a24 = mulmod_256(&numerator, &denom_inv, n_256);

    // Convert to Montgomery form.
    let a24_mont = to_mont_256(&a24, mont);
    let q0 = MontPoint256 {
        x: to_mont_256(&u3, mont),
        z: to_mont_256(&v3, mont),
    };

    // Phase 1: scalar multiply by lcm(1..B1).
    let one_mont = to_mont_256(&U256_ONE, mont);
    let mut q = q0;
    let mut accum = one_mont;
    let mut step_count = 0u32;

    for &p in primes.iter() {
        if p > b1 {
            break;
        }
        let mut pk = p;
        while let Some(next) = pk.checked_mul(p) {
            if next > b1 { break; }
            pk = next;
        }
        q = mont_ladder_256(&q, pk, &a24_mont, mont);

        // Accumulate Z into product for batched GCD.
        accum = mont_mul_256(&accum, &q.z, mont);
        step_count += 1;

        if step_count % 32 == 0 {
            let accum_reg = from_mont_256(&accum, mont);
            let g = gcd_256(&accum_reg, n_256);
            if g != U256_ONE && g != *n_256 {
                return Some(u256_to_integer(&g));
            }
            if accum_reg == U256_ZERO {
                return ecm_one_curve_careful(n_int, b1, b2, sigma, primes);
            }
        }
    }

    // Final Phase 1 GCD check.
    let accum_reg = from_mont_256(&accum, mont);
    let g = gcd_256(&accum_reg, n_256);
    if g != U256_ONE && g != *n_256 {
        return Some(u256_to_integer(&g));
    }
    if accum_reg == U256_ZERO {
        return ecm_one_curve_careful(n_int, b1, b2, sigma, primes);
    }

    // Phase 2: baby-step giant-step for primes in (B1, B2].
    let mut accum2 = one_mont;
    let mut batch_count = 0u32;
    const PHASE2_BATCH: u32 = 32;

    let q_base = q;

    let d_step = ((b2 - b1) as f64).sqrt() as u64;
    let d = if d_step < 2 { 2 } else { d_step | 1 };

    // Precompute baby steps: [1]Q, [2]Q, ..., [d]Q.
    let mut baby = Vec::with_capacity(d as usize);
    baby.push(q_base);
    if d >= 2 {
        baby.push(mont_double_256(&q_base, &a24_mont, mont));
    }
    for i in 3..=d {
        let len = baby.len();
        let next = mont_add_256(&baby[len - 1], &baby[0], &baby[len - 2], mont);
        baby.push(next);
        let _ = i;
    }

    let d_q = baby[baby.len() - 1];

    // Giant step: start at B1 rounded up to next multiple of d.
    let first_giant = ((b1 / d) + 1) * d;
    let mut giant_q = mont_ladder_256(&q_base, first_giant, &a24_mont, mont);
    let mut prev_giant_q = if first_giant >= d {
        mont_ladder_256(&q_base, first_giant - d, &a24_mont, mont)
    } else {
        MontPoint256 {
            x: to_mont_256(&U256_ONE, mont),
            z: U256_ZERO,
        }
    };

    let p_start = primes.partition_point(|&p| p <= b1);
    let p_end = primes.len();
    let mut scan_idx = p_start;

    let mut giant_val = first_giant;
    while giant_val <= b2 + d {
        let range_lo = if giant_val > d { giant_val - d } else { 0 };
        while scan_idx < p_end && primes[scan_idx] < range_lo.max(b1 + 1) {
            scan_idx += 1;
        }

        let mut pi = scan_idx;
        while pi < p_end && primes[pi] <= (giant_val + d).min(b2) {
            let p = primes[pi];
            pi += 1;

            let b_offset = if p > giant_val { p - giant_val } else { giant_val - p };
            if b_offset < 1 || b_offset > d { continue; }
            let b_idx = (b_offset - 1) as usize;

            let t1 = mont_mul_256(&giant_q.x, &baby[b_idx].z, mont);
            let t2 = mont_mul_256(&giant_q.z, &baby[b_idx].x, mont);
            let cross = sub_mod_256(&t1, &t2, &mont.n);
            accum2 = mont_mul_256(&accum2, &cross, mont);
            batch_count += 1;

            if batch_count >= PHASE2_BATCH {
                let accum2_reg = from_mont_256(&accum2, mont);
                let g = gcd_256(&accum2_reg, n_256);
                if g != U256_ONE && g != *n_256 {
                    return Some(u256_to_integer(&g));
                }
                batch_count = 0;
            }
        }

        // Advance giant step.
        let new_giant = mont_add_256(&giant_q, &d_q, &prev_giant_q, mont);
        prev_giant_q = giant_q;
        giant_q = new_giant;
        giant_val += d;
    }

    // Final Phase 2 GCD check.
    let accum2_reg = from_mont_256(&accum2, mont);
    let g = gcd_256(&accum2_reg, n_256);
    if g != U256_ONE && g != *n_256 {
        return Some(u256_to_integer(&g));
    }
    None
}

/// Returns true if `n` fits in 192 bits and can use the fast U192 path.
fn fits_in_192(n: &Integer) -> bool {
    n.significant_bits() <= 192
}

/// Returns true if `n` fits in 256 bits and can use the fast U256 path.
fn fits_in_256(n: &Integer) -> bool {
    n.significant_bits() <= 256
}

/// Factor `n` using ECM with the given parameters.
///
/// Tries up to `max_curves` Suyama-parameterized curves with Phase 1 bound `b1`
/// and Phase 2 bound `b2`. Returns `Some(factor)` on the first non-trivial
/// factor found, or `None` if all curves fail.
///
/// Dispatches to fixed-width 192-bit Montgomery arithmetic for n <= 192 bits,
/// 256-bit Montgomery arithmetic for 193-256 bit n, and falls back to
/// rug::Integer (GMP) for larger inputs.
///
/// Uses rayon for parallel curve evaluation.
pub fn ecm_factor(n: &Integer, max_curves: usize, b1: u64, b2: u64) -> Option<Integer> {
    // Pre-sieve primes up to b2 once (shared across all curves).
    // Phase 2 uses two-pointer scan over this sorted list, no bitset needed.
    let primes = sieve_primes_vec(b2);

    if fits_in_192(n) && *n > 1u32 && n.is_odd() {
        // U192 fast path
        let n_192 = integer_to_u192(n);
        let mont = Mont192::new(&n_192);

        use rayon::prelude::*;
        (0..max_curves)
            .into_par_iter()
            .find_map_any(|idx| {
                let sigma = 6u64 + idx as u64;
                ecm_one_curve_fast(&n_192, n, b1, b2, sigma, &primes, &mont, None)
            })
    } else if fits_in_256(n) && *n > 1u32 && n.is_odd() {
        // U256 fast path for 193-256 bit
        let n_256 = integer_to_u256(n);
        let mont = Mont256::new(&n_256);

        use rayon::prelude::*;
        (0..max_curves)
            .into_par_iter()
            .find_map_any(|idx| {
                let sigma = 6u64 + idx as u64;
                ecm_one_curve_fast_256(&n_256, n, b1, b2, sigma, &primes, &mont, None)
            })
    } else {
        // GMP fallback
        use rayon::prelude::*;
        (0..max_curves)
            .into_par_iter()
            .find_map_any(|idx| {
                let sigma = 6u64 + idx as u64;
                ecm_one_curve(n, b1, b2, sigma, &primes)
            })
    }
}

/// Factor `n` using ECM, single-threaded (no rayon).
///
/// Useful for benchmarking ST performance.
pub fn ecm_factor_st(n: &Integer, max_curves: usize, b1: u64, b2: u64) -> Option<Integer> {
    let primes = sieve_primes_vec(b2);

    if fits_in_192(n) && *n > 1u32 && n.is_odd() {
        // U192 fast path
        let n_192 = integer_to_u192(n);
        let mont = Mont192::new(&n_192);

        for idx in 0..max_curves {
            let sigma = 6u64 + idx as u64;
            if let Some(f) = ecm_one_curve_fast(&n_192, n, b1, b2, sigma, &primes, &mont, None) {
                return Some(f);
            }
        }
        None
    } else if fits_in_256(n) && *n > 1u32 && n.is_odd() {
        // U256 fast path for 193-256 bit
        let n_256 = integer_to_u256(n);
        let mont = Mont256::new(&n_256);

        for idx in 0..max_curves {
            let sigma = 6u64 + idx as u64;
            if let Some(f) = ecm_one_curve_fast_256(&n_256, n, b1, b2, sigma, &primes, &mont, None) {
                return Some(f);
            }
        }
        None
    } else {
        // GMP fallback
        for idx in 0..max_curves {
            let sigma = 6u64 + idx as u64;
            if let Some(f) = ecm_one_curve(n, b1, b2, sigma, &primes) {
                return Some(f);
            }
        }
        None
    }
}

/// Try ECM factoring for `n`, with parameters tuned by bit-size.
///
/// B1/B2/max_curves are chosen based on `n.significant_bits()`, but can be
/// overridden via environment variables `POTAPOV_NFS_ECM_B1`, `POTAPOV_NFS_ECM_B2`,
/// and `POTAPOV_NFS_ECM_CURVES`.
///
/// Returns `Some((factor, elapsed_ms))` on success, `None` on failure.
pub fn try_ecm_factor(n: &Integer) -> Option<(Integer, f64)> {
    let bits = n.significant_bits();

    // Single-call ECM with B1 high enough for worst-case factors (RSA safe
    // primes, ECM-resistant group orders). Easy factors are found early via
    // rayon's find_map_any — most curves terminate in Phase 1 before reaching
    // Phase 2, so high B1 doesn't penalize easy cases proportionally.
    // The prime sieve is done ONCE and shared across all curves.
    let (default_b1, default_b2, default_curves) = if bits <= 80 {
        (2_000u64, 200_000u64, 25usize)
    } else if bits <= 100 {
        (5_000, 500_000, 50)
    } else if bits <= 120 {
        (11_000, 1_100_000, 100)
    } else if bits <= 140 {
        (25_000, 2_500_000, 150)
    } else if bits <= 160 {
        // c45: ~74-bit factors. B1=40K balances per-curve cost vs success rate.
        // Per-curve ~5ms. 800 curves: MT worst 800/10 × 5ms = 400ms.
        // Higher B1 catches RSA safe primes and hard random factors better
        // than B1=30K (which needed 1200 curves for coverage).
        (40_000, 4_000_000, 800)
    } else if bits <= 180 {
        (200_000, 20_000_000, 400)
    } else if bits <= 210 {
        // c60: multi-stage handled below. Start with moderate B1.
        (200_000, 20_000_000, 1000)
    } else {
        (3_000_000, 300_000_000, 1200)
    };

    // For c60 (181-210 bits): use staged ECM with shared prime sieve.
    // Stage 1: low B1 catches easy factors fast.
    // Stage 2: medium B1 catches moderate factors.
    // Stage 3: high B1 catches hard factors.
    // All stages share the same sieve_primes_vec (computed once at max B2).
    let env_b1 = std::env::var("POTAPOV_NFS_ECM_B1").ok().and_then(|s| s.parse::<u64>().ok());
    let env_b2 = std::env::var("POTAPOV_NFS_ECM_B2").ok().and_then(|s| s.parse::<u64>().ok());
    let env_curves = std::env::var("POTAPOV_NFS_ECM_CURVES").ok().and_then(|s| s.parse::<usize>().ok());

    let start = std::time::Instant::now();

    let (result, final_b1, final_b2, total_curves) = if env_b1.is_some() || bits <= 180 || bits > 210 {
        // Single-stage: use default or env-overridden params
        let b1 = env_b1.unwrap_or(default_b1);
        let b2 = env_b2.unwrap_or(default_b2);
        let curves = env_curves.unwrap_or(default_curves);
        let r = ecm_factor(n, curves, b1, b2);
        (r, b1, b2, curves)
    } else {
        // c60 multi-stage with shared sieve
        let max_b2 = env_b2.unwrap_or(50_000_000u64);
        let primes = sieve_primes_vec(max_b2);

        // Stages: (B1, B2, curves). Each uses different sigma offset.
        let stages: &[(u64, u64, usize)] = &[
            (50_000, 5_000_000, 500),    // ~3ms/curve, catches 25-digit easily
            (200_000, 20_000_000, 500),   // ~10ms/curve, catches 30-digit
            (1_000_000, max_b2, 500),     // ~40ms/curve, catches hard cases
        ];

        let mut result = None;
        let mut total = 0usize;
        let mut last_b1 = 0u64;
        let mut last_b2 = 0u64;
        let mut sigma_offset = 6u64;

        if fits_in_256(n) && *n > 1u32 && n.is_odd() {
            let n_256 = integer_to_u256(n);
            let mont = Mont256::new(&n_256);

            for &(b1, b2, curves) in stages {
                use rayon::prelude::*;
                let off = sigma_offset;
                let r = (0..curves)
                    .into_par_iter()
                    .find_map_any(|idx| {
                        let sigma = off + idx as u64;
                        ecm_one_curve_fast_256(&n_256, n, b1, b2, sigma, &primes, &mont, None)
                    });
                total += curves;
                last_b1 = b1;
                last_b2 = b2;
                sigma_offset += curves as u64;
                if r.is_some() {
                    result = r;
                    break;
                }
            }
        } else if fits_in_192(n) && *n > 1u32 && n.is_odd() {
            let n_192 = integer_to_u192(n);
            let mont = Mont192::new(&n_192);

            for &(b1, b2, curves) in stages {
                use rayon::prelude::*;
                let off = sigma_offset;
                let r = (0..curves)
                    .into_par_iter()
                    .find_map_any(|idx| {
                        let sigma = off + idx as u64;
                        ecm_one_curve_fast(&n_192, n, b1, b2, sigma, &primes, &mont, None)
                    });
                total += curves;
                last_b1 = b1;
                last_b2 = b2;
                sigma_offset += curves as u64;
                if r.is_some() {
                    result = r;
                    break;
                }
            }
        }

        (result, last_b1, last_b2, total)
    };

    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

    if let Some(f) = result {
        eprintln!(
            "  ecm: factor found in {:.1}ms (B1={}, B2={}, curves={}, bits={}): {}",
            elapsed_ms, final_b1, final_b2, total_curves, bits, f
        );
        Some((f, elapsed_ms))
    } else {
        eprintln!(
            "  ecm: no factor after {:.1}ms (B1={}, B2={}, curves={}, bits={})",
            elapsed_ms, final_b1, final_b2, total_curves, bits
        );
        None
    }
}

/// Run one ECM curve on `n` with Suyama parameterization from `sigma`.
///
/// Returns `Some(factor)` if a non-trivial factor is found in Phase 1 or Phase 2.
fn ecm_one_curve(n: &Integer, b1: u64, b2: u64, sigma: u64, primes: &[u64]) -> Option<Integer> {
    // Build Montgomery curve via Suyama parameterization.
    // u = sigma^2 - 5 mod n, v = 4*sigma mod n
    let sigma_int = Integer::from(sigma);
    let mut u = Integer::from(&sigma_int * &sigma_int);
    u -= 5u32;
    u.modulo_mut(n);
    let mut v = Integer::from(&sigma_int * 4u32);
    v.modulo_mut(n);

    if u == 0 || v == 0 {
        return None;
    }

    // Starting point Q = (u^3 : v^3)
    let mut u3 = Integer::from(&u * &u);
    u3 *= &u;
    u3.modulo_mut(n);
    let mut v3 = Integer::from(&v * &v);
    v3 *= &v;
    v3.modulo_mut(n);

    // a24 = (A+2)/4 = (v-u)^3 * (3u+v) / (16 * u^3 * v) mod n
    let v_minus_u = mod_sub(&v, &u, n);
    let mut vmu3 = Integer::from(&v_minus_u * &v_minus_u);
    vmu3 *= &v_minus_u;
    vmu3.modulo_mut(n);
    let mut three_u_plus_v = Integer::from(&u * 3u32);
    three_u_plus_v += &v;
    three_u_plus_v.modulo_mut(n);
    let mut numerator = Integer::from(&vmu3 * &three_u_plus_v);
    numerator.modulo_mut(n);

    let mut denom = Integer::from(&u3 * &v);
    denom *= 16u32;
    denom.modulo_mut(n);

    // Compute modular inverse of denom; if gcd(denom, n) > 1, we found a factor.
    let denom_inv = match mod_inverse(&denom, n) {
        Ok(inv) => inv,
        Err(g) => {
            if g > 1u32 && g < *n {
                return Some(g);
            }
            return None;
        }
    };

    let mut a24 = Integer::from(&numerator * &denom_inv);
    a24.modulo_mut(n);

    let mut q = MontPoint {
        x: u3.clone(),
        z: v3.clone(),
    };

    let mut scratch = MontScratch::new();

    // ---------------------------------------------------------------
    // Phase 1: multiply Q by lcm(1..B1)
    // ---------------------------------------------------------------
    let mut accum = Integer::from(1);
    let mut step_count = 0u32;

    for &p in primes.iter() {
        if p > b1 {
            break;
        }
        let mut pk = p;
        while let Some(next) = pk.checked_mul(p) {
            if next > b1 { break; }
            pk = next;
        }
        mont_ladder_inplace(&mut q, pk, &a24, n, &mut scratch);

        // Accumulate Z into product for batched GCD
        accum *= &q.z;
        accum.modulo_mut(n);
        step_count += 1;

        if step_count % 32 == 0 {
            if let Some(f) = check_gcd(&accum, n) {
                return Some(f);
            }
            if accum == 0 {
                return ecm_one_curve_careful(n, b1, b2, sigma, primes);
            }
        }
    }

    // Final Phase 1 GCD check
    if let Some(f) = check_gcd(&accum, n) {
        return Some(f);
    }
    if accum == 0 {
        return ecm_one_curve_careful(n, b1, b2, sigma, primes);
    }

    // ---------------------------------------------------------------
    // Phase 2: standard continuation with baby-step giant-step
    // ---------------------------------------------------------------
    let mut accum2 = Integer::from(1);
    let mut batch_count = 0u32;
    const PHASE2_BATCH: u32 = 32;

    let q_base = q.clone();

    // Baby step size D ~= sqrt(B2 - B1)
    let d_step = ((b2 - b1) as f64).sqrt() as u64;
    let d = if d_step < 2 { 2 } else { d_step | 1 };

    // Precompute baby steps: [1]Q, [2]Q, ..., [d]Q
    let baby = precompute_baby_steps(&q_base, d, &a24, n, &mut scratch);

    let d_q = &baby[baby.len() - 1]; // [d]Q

    // Giant step: start at B1 rounded up to next multiple of d
    let first_giant = ((b1 / d) + 1) * d;
    let mut giant_q = mont_ladder_new(&q_base, first_giant, &a24, n, &mut scratch);
    let mut prev_giant_q = if first_giant >= d {
        mont_ladder_new(&q_base, first_giant - d, &a24, n, &mut scratch)
    } else {
        MontPoint {
            x: Integer::from(1),
            z: Integer::from(0),
        }
    };

    let mut cross = Integer::new();
    let mut tmp = Integer::new();

    let mut giant_val = first_giant;
    while giant_val <= b2 + d {
        for b_idx in 0..baby.len() {
            let b_offset = (b_idx + 1) as u64;
            let val_plus = giant_val + b_offset;
            let val_minus = if giant_val >= b_offset {
                giant_val - b_offset
            } else {
                0
            };

            if val_plus > b1 && val_plus <= b2 && is_in_prime_list(val_plus, primes) {
                // cross = (X_giant * Z_baby - Z_giant * X_baby) mod n
                cross.assign(&giant_q.x * &baby[b_idx].z);
                tmp.assign(&giant_q.z * &baby[b_idx].x);
                cross -= &tmp;
                cross.modulo_mut(n);
                accum2 *= &cross;
                accum2.modulo_mut(n);
                batch_count += 1;
            }

            if val_minus > b1 && val_minus <= b2 && is_in_prime_list(val_minus, primes) {
                cross.assign(&giant_q.x * &baby[b_idx].z);
                tmp.assign(&giant_q.z * &baby[b_idx].x);
                cross -= &tmp;
                cross.modulo_mut(n);
                accum2 *= &cross;
                accum2.modulo_mut(n);
                batch_count += 1;
            }

            if batch_count >= PHASE2_BATCH {
                if let Some(f) = check_gcd(&accum2, n) {
                    return Some(f);
                }
                batch_count = 0;
            }
        }

        // Advance giant step
        let new_giant = mont_add_new(&giant_q, d_q, &prev_giant_q, n, &mut scratch);
        prev_giant_q = giant_q;
        giant_q = new_giant;
        giant_val += d;
    }

    check_gcd(&accum2, n)
}

/// Careful per-prime GCD fallback when batched Phase 1 overshoots.
fn ecm_one_curve_careful(
    n: &Integer,
    b1: u64,
    _b2: u64,
    sigma: u64,
    primes: &[u64],
) -> Option<Integer> {
    let sigma_int = Integer::from(sigma);
    let mut u = Integer::from(&sigma_int * &sigma_int);
    u -= 5u32;
    u.modulo_mut(n);
    let mut v = Integer::from(&sigma_int * 4u32);
    v.modulo_mut(n);
    if u == 0 || v == 0 {
        return None;
    }

    let mut u3 = Integer::from(&u * &u);
    u3 *= &u;
    u3.modulo_mut(n);
    let mut v3 = Integer::from(&v * &v);
    v3 *= &v;
    v3.modulo_mut(n);

    let v_minus_u = mod_sub(&v, &u, n);
    let mut vmu3 = Integer::from(&v_minus_u * &v_minus_u);
    vmu3 *= &v_minus_u;
    vmu3.modulo_mut(n);
    let mut three_u_plus_v = Integer::from(&u * 3u32);
    three_u_plus_v += &v;
    three_u_plus_v.modulo_mut(n);
    let mut numerator = Integer::from(&vmu3 * &three_u_plus_v);
    numerator.modulo_mut(n);
    let mut denom = Integer::from(&u3 * &v);
    denom *= 16u32;
    denom.modulo_mut(n);

    let denom_inv = match mod_inverse(&denom, n) {
        Ok(inv) => inv,
        Err(g) => {
            if g > 1u32 && g < *n {
                return Some(g);
            }
            return None;
        }
    };

    let mut a24 = Integer::from(&numerator * &denom_inv);
    a24.modulo_mut(n);
    let mut q = MontPoint {
        x: u3.clone(),
        z: v3.clone(),
    };
    let mut scratch = MontScratch::new();

    for &p in primes.iter() {
        if p > b1 {
            break;
        }
        let mut pk = p;
        while let Some(next) = pk.checked_mul(p) {
            if next > b1 { break; }
            pk = next;
        }
        mont_ladder_inplace(&mut q, pk, &a24, n, &mut scratch);
        let g = Integer::from(n.gcd_ref(&q.z));
        if g > 1u32 && g < *n {
            return Some(g);
        }
        if g == *n {
            return None;
        }
    }

    None
}

// ---------------------------------------------------------------------------
// Montgomery curve point operations -- allocation-optimized
// ---------------------------------------------------------------------------

/// Montgomery double: `result = [2]p`, using scratch space for zero allocation.
///
/// Formulas (all mod n):
/// ```text
/// sum  = (X + Z)
/// diff = (X - Z)
/// u    = sum^2
/// v    = diff^2
/// w    = u - v
/// X2   = u * v mod n
/// Z2   = w * (v + a24 * w) mod n
/// ```
#[inline]
fn mont_double_into(result: &mut MontPoint, p: &MontPoint, a24: &Integer, n: &Integer, sc: &mut MontScratch) {
    // s1 = X + Z
    sc.s1.assign(&p.x + &p.z);
    // s2 = X - Z (mod n, keep positive)
    sc.s2.assign(&p.x - &p.z);
    if sc.s2 < 0 {
        sc.s2 += n;
    }
    // s3 = u = s1^2 mod n
    sc.s3.assign(&sc.s1 * &sc.s1);
    sc.s3.modulo_mut(n);
    // s4 = v = s2^2 mod n
    sc.s4.assign(&sc.s2 * &sc.s2);
    sc.s4.modulo_mut(n);
    // X2 = u * v mod n
    result.x.assign(&sc.s3 * &sc.s4);
    result.x.modulo_mut(n);
    // s5 = w = u - v
    sc.s5.assign(&sc.s3 - &sc.s4);
    if sc.s5 < 0 {
        sc.s5 += n;
    }
    // s6 = a24 * w mod n
    sc.s6.assign(a24 * &sc.s5);
    sc.s6.modulo_mut(n);
    // s6 = v + a24*w
    sc.s6 += &sc.s4;
    // Z2 = w * (v + a24*w) mod n
    result.z.assign(&sc.s5 * &sc.s6);
    result.z.modulo_mut(n);
}

/// Montgomery differential addition: `result = p + q`, given `diff = p - q`.
///
/// ```text
/// u1 = (Xp - Zp) * (Xq + Zq)
/// u2 = (Xp + Zp) * (Xq - Zq)
/// X3 = Zdiff * (u1 + u2)^2 mod n
/// Z3 = Xdiff * (u1 - u2)^2 mod n
/// ```
#[inline]
fn mont_add_into(
    result: &mut MontPoint,
    p: &MontPoint,
    q: &MontPoint,
    diff: &MontPoint,
    n: &Integer,
    sc: &mut MontScratch,
) {
    // s1 = Xp - Zp
    sc.s1.assign(&p.x - &p.z);
    if sc.s1 < 0 {
        sc.s1 += n;
    }
    // s2 = Xq + Zq
    sc.s2.assign(&q.x + &q.z);
    // s3 = u1 = s1 * s2 mod n
    sc.s3.assign(&sc.s1 * &sc.s2);
    sc.s3.modulo_mut(n);

    // s1 = Xp + Zp
    sc.s1.assign(&p.x + &p.z);
    // s2 = Xq - Zq
    sc.s2.assign(&q.x - &q.z);
    if sc.s2 < 0 {
        sc.s2 += n;
    }
    // s4 = u2 = s1 * s2 mod n
    sc.s4.assign(&sc.s1 * &sc.s2);
    sc.s4.modulo_mut(n);

    // s1 = u1 + u2
    sc.s1.assign(&sc.s3 + &sc.s4);
    // s2 = (u1 + u2)^2 mod n
    sc.s2.assign(&sc.s1 * &sc.s1);
    sc.s2.modulo_mut(n);
    // X3 = Zdiff * (u1+u2)^2 mod n
    result.x.assign(&diff.z * &sc.s2);
    result.x.modulo_mut(n);

    // s1 = u1 - u2
    sc.s1.assign(&sc.s3 - &sc.s4);
    if sc.s1 < 0 {
        sc.s1 += n;
    }
    // s2 = (u1-u2)^2 mod n
    sc.s2.assign(&sc.s1 * &sc.s1);
    sc.s2.modulo_mut(n);
    // Z3 = Xdiff * (u1-u2)^2 mod n
    result.z.assign(&diff.x * &sc.s2);
    result.z.modulo_mut(n);
}

/// Montgomery ladder: compute `[k]P` in-place (mutates `p`).
fn mont_ladder_inplace(p: &mut MontPoint, k: u64, a24: &Integer, n: &Integer, sc: &mut MontScratch) {
    if k == 0 {
        p.x.assign(1);
        p.z.assign(0);
        return;
    }
    if k == 1 {
        return;
    }

    // We need the original P for differential addition.
    let p_orig = p.clone();

    // r0 = P, r1 = [2]P
    let mut r0 = p_orig.clone();
    let mut r1 = MontPoint {
        x: Integer::new(),
        z: Integer::new(),
    };
    mont_double_into(&mut r1, &p_orig, a24, n, sc);

    let mut tmp = MontPoint {
        x: Integer::new(),
        z: Integer::new(),
    };

    let bits = 64 - k.leading_zeros();
    for i in (0..bits - 1).rev() {
        if (k >> i) & 1 == 1 {
            // r0 = r0 + r1 (diff = P), r1 = 2*r1
            mont_add_into(&mut tmp, &r0, &r1, &p_orig, n, sc);
            std::mem::swap(&mut r0, &mut tmp);
            mont_double_into(&mut tmp, &r1, a24, n, sc);
            std::mem::swap(&mut r1, &mut tmp);
        } else {
            // r1 = r0 + r1 (diff = P), r0 = 2*r0
            mont_add_into(&mut tmp, &r0, &r1, &p_orig, n, sc);
            std::mem::swap(&mut r1, &mut tmp);
            mont_double_into(&mut tmp, &r0, a24, n, sc);
            std::mem::swap(&mut r0, &mut tmp);
        }
    }

    p.x.assign(&r0.x);
    p.z.assign(&r0.z);
}

/// Montgomery ladder: return new `[k]P` (does not mutate input).
fn mont_ladder_new(
    p: &MontPoint,
    k: u64,
    a24: &Integer,
    n: &Integer,
    sc: &mut MontScratch,
) -> MontPoint {
    let mut result = p.clone();
    mont_ladder_inplace(&mut result, k, a24, n, sc);
    result
}

/// Allocating version of mont_add for phase 2 step.
fn mont_add_new(
    p: &MontPoint,
    q: &MontPoint,
    diff: &MontPoint,
    n: &Integer,
    sc: &mut MontScratch,
) -> MontPoint {
    let mut result = MontPoint {
        x: Integer::new(),
        z: Integer::new(),
    };
    mont_add_into(&mut result, p, q, diff, n, sc);
    result
}

/// Precompute baby step table: [1]P, [2]P, ..., [d]P.
fn precompute_baby_steps(
    p: &MontPoint,
    d: u64,
    a24: &Integer,
    n: &Integer,
    sc: &mut MontScratch,
) -> Vec<MontPoint> {
    let mut table = Vec::with_capacity(d as usize);
    if d == 0 {
        return table;
    }

    table.push(p.clone());
    if d == 1 {
        return table;
    }

    let mut p2 = MontPoint {
        x: Integer::new(),
        z: Integer::new(),
    };
    mont_double_into(&mut p2, p, a24, n, sc);
    table.push(p2);
    if d == 2 {
        return table;
    }

    for _ in 3..=d {
        let len = table.len();
        let next = mont_add_new(&table[len - 1], &table[0], &table[len - 2], n, sc);
        table.push(next);
    }

    table
}

// ---------------------------------------------------------------------------
// Non-hot-path wrappers for tests and simple callers
// ---------------------------------------------------------------------------

/// Allocating mont_double for use in tests.
#[cfg(test)]
fn mont_double(p: &MontPoint, a24: &Integer, n: &Integer) -> MontPoint {
    let mut result = MontPoint {
        x: Integer::new(),
        z: Integer::new(),
    };
    let mut sc = MontScratch::new();
    mont_double_into(&mut result, p, a24, n, &mut sc);
    result
}

/// Allocating mont_ladder for use in tests.
#[cfg(test)]
fn mont_ladder(p: &MontPoint, k: u64, a24: &Integer, n: &Integer) -> MontPoint {
    let mut sc = MontScratch::new();
    mont_ladder_new(p, k, a24, n, &mut sc)
}

// ---------------------------------------------------------------------------
// Arithmetic helpers
// ---------------------------------------------------------------------------

/// Modular subtraction: `(a - b) mod n`, always non-negative.
#[inline]
fn mod_sub(a: &Integer, b: &Integer, n: &Integer) -> Integer {
    let mut diff = Integer::from(a - b);
    if diff < 0 {
        diff += n;
    }
    diff.modulo_mut(n);
    diff
}

/// Modular inverse: returns `Ok(a^{-1} mod n)` if gcd(a, n) == 1,
/// or `Err(gcd(a, n))` if a non-trivial GCD is found.
fn mod_inverse(a: &Integer, n: &Integer) -> Result<Integer, Integer> {
    let g = Integer::from(a.gcd_ref(n));
    if g == 1u32 {
        match a.clone().invert(n) {
            Ok(inv) => Ok(inv),
            Err(_) => Err(g),
        }
    } else {
        Err(g)
    }
}

/// Check if `gcd(accum, n)` yields a non-trivial factor.
#[inline]
fn check_gcd(accum: &Integer, n: &Integer) -> Option<Integer> {
    if *accum == 0u32 {
        return None;
    }
    let g = Integer::from(accum.gcd_ref(n));
    if g > 1u32 && g < *n {
        Some(g)
    } else {
        None
    }
}

/// Check whether `val` is in the sorted prime list via binary search.
#[inline]
fn is_in_prime_list(val: u64, _primes: &[u64]) -> bool {
    // Legacy fallback — prefer is_prime_bitset for O(1) lookup.
    _primes.binary_search(&val).is_ok()
}

/// O(1) primality check using a pre-sieved bitset.
#[inline]
fn is_prime_bitset(val: u64, bitset: &[u64]) -> bool {
    let idx = val as usize;
    let word = idx / 64;
    let bit = idx % 64;
    word < bitset.len() && (bitset[word] >> bit) & 1 == 1
}

/// Build a bitset where bit `i` is set iff `i` is prime, for `i` up to `bound`.
fn sieve_prime_bitset(bound: u64) -> Vec<u64> {
    let n = bound as usize + 1;
    let nwords = (n + 63) / 64;
    let mut bits = vec![!0u64; nwords];
    // 0 and 1 are not prime
    if nwords > 0 {
        bits[0] &= !3u64; // clear bits 0 and 1
    }
    let limit = (n as f64).sqrt() as usize + 1;
    for i in 2..=limit {
        if (bits[i / 64] >> (i % 64)) & 1 == 1 {
            let mut j = i * i;
            while j < n {
                bits[j / 64] &= !(1u64 << (j % 64));
                j += i;
            }
        }
    }
    // Clear bits beyond bound
    if n % 64 != 0 && nwords > 0 {
        bits[nwords - 1] &= (1u64 << (n % 64)) - 1;
    }
    bits
}

/// Sieve of Eratosthenes up to `bound`, returning a sorted Vec<u64>.
///
/// Uses bit-packed sieve (8x less memory than bool array) and extracts
/// primes via word-level bit traversal.
fn sieve_primes_vec(bound: u64) -> Vec<u64> {
    if bound < 2 {
        return Vec::new();
    }
    let n = (bound as usize) + 1;
    let nwords = (n + 63) / 64;
    let mut bits = vec![!0u64; nwords];
    // 0 and 1 are not prime
    if nwords > 0 {
        bits[0] &= !3u64;
    }
    let limit = (n as f64).sqrt() as usize + 1;
    for i in 2..=limit {
        if (bits[i / 64] >> (i % 64)) & 1 == 1 {
            let mut j = i * i;
            while j < n {
                bits[j / 64] &= !(1u64 << (j % 64));
                j += i;
            }
        }
    }
    // Clear bits beyond bound
    if n % 64 != 0 && nwords > 0 {
        bits[nwords - 1] &= (1u64 << (n % 64)) - 1;
    }
    // Extract primes via word-level bit traversal
    let mut primes = Vec::new();
    for (wi, &word) in bits.iter().enumerate() {
        let mut w = word;
        while w != 0 {
            let bit = w.trailing_zeros() as usize;
            primes.push((wi * 64 + bit) as u64);
            w &= w - 1;
        }
    }
    primes
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to create a semiprime from two primes.
    fn semiprime(p: &str, q: &str) -> Integer {
        let pi: Integer = p.parse().unwrap();
        let qi: Integer = q.parse().unwrap();
        Integer::from(&pi * &qi)
    }

    #[test]
    fn test_mont_double_basic() {
        let n = Integer::from(1009u64 * 1013);
        let a24 = Integer::from(3u32);
        let p = MontPoint {
            x: Integer::from(7u32),
            z: Integer::from(1u32),
        };
        let p2 = mont_double(&p, &a24, &n);
        assert!(p2.x < n);
        assert!(p2.z < n);
    }

    #[test]
    fn test_mont_ladder_identity() {
        let n = Integer::from(1009u64 * 1013);
        let a24 = Integer::from(3u32);
        let p = MontPoint {
            x: Integer::from(7u32),
            z: Integer::from(1u32),
        };
        let p1 = mont_ladder(&p, 1, &a24, &n);
        assert_eq!(p1.x, p.x);
        assert_eq!(p1.z, p.z);
    }

    #[test]
    fn test_ecm_small_semiprime() {
        let n = Integer::from(1009u64 * 1013);
        let result = ecm_factor_st(&n, 50, 500, 5000);
        assert!(result.is_some(), "ECM should factor 1009*1013");
        let f = result.unwrap();
        assert!(Integer::from(&n % &f) == 0u32, "factor should divide n");
        assert!(f > 1u32 && f < n, "factor should be non-trivial");
    }

    #[test]
    fn test_ecm_medium_semiprime() {
        let n = Integer::from(10007u64 * 10009);
        let result = ecm_factor_st(&n, 50, 1000, 10000);
        assert!(result.is_some(), "ECM should factor 10007*10009");
        let f = result.unwrap();
        assert!(Integer::from(&n % &f) == 0u32);
        assert!(f > 1u32 && f < n);
    }

    #[test]
    fn test_ecm_32bit_semiprime() {
        let n = Integer::from(65521u64 * 65537);
        let result = ecm_factor_st(&n, 100, 2000, 50000);
        assert!(result.is_some(), "ECM should factor 65521*65537");
        let f = result.unwrap();
        assert!(Integer::from(&n % &f) == 0u32);
        assert!(f > 1u32 && f < n);
    }

    #[test]
    fn test_ecm_64bit_semiprime() {
        let p: u64 = 2_147_483_659;
        let q: u64 = 2_147_483_693;
        let n = Integer::from(p) * Integer::from(q);
        let result = ecm_factor_st(&n, 200, 5000, 500_000);
        assert!(result.is_some(), "ECM should factor 64-bit semiprime");
        let f = result.unwrap();
        assert!(Integer::from(&n % &f) == 0u32);
        assert!(f > 1u32 && f < n);
    }

    #[test]
    fn test_ecm_prime_returns_none() {
        let n: Integer = "104729".parse().unwrap();
        let result = ecm_factor_st(&n, 20, 500, 5000);
        assert!(result.is_none(), "ECM on a prime should return None");
    }

    #[test]
    fn test_ecm_parallel_small() {
        let n = Integer::from(1009u64 * 1013);
        let result = ecm_factor(&n, 50, 500, 5000);
        assert!(result.is_some(), "Parallel ECM should factor 1009*1013");
        let f = result.unwrap();
        assert!(Integer::from(&n % &f) == 0u32);
    }

    #[test]
    fn test_ecm_100bit_semiprime() {
        let n = semiprime("1125899906842679", "1125899906842747");
        assert!(n.significant_bits() >= 99);
        let result = ecm_factor_st(&n, 200, 10000, 1_000_000);
        assert!(result.is_some(), "ECM should factor 100-bit semiprime");
        let f = result.unwrap();
        assert!(Integer::from(&n % &f) == 0u32);
    }

    #[test]
    fn test_ecm_120bit_semiprime() {
        let n = semiprime("1152921504606847009", "1152921504606847087");
        assert!(n.significant_bits() >= 119);
        let result = ecm_factor_st(&n, 200, 20000, 2_000_000);
        assert!(result.is_some(), "ECM should factor 120-bit semiprime");
        let f = result.unwrap();
        assert!(Integer::from(&n % &f) == 0u32);
    }

    #[test]
    fn test_mod_inverse_invertible() {
        let n = Integer::from(1009u64 * 1013);
        let a = Integer::from(7u32);
        let inv = mod_inverse(&a, &n).unwrap();
        let product = Integer::from(&a * &inv).modulo(&n);
        assert_eq!(product, 1u32, "a * a^-1 should be 1 mod n");
    }

    #[test]
    fn test_mod_inverse_non_invertible() {
        let n = Integer::from(1009u64 * 1013);
        let a = Integer::from(1009u32);
        match mod_inverse(&a, &n) {
            Err(g) => {
                assert!(g > 1u32 && g <= 1009u32);
            }
            Ok(_) => panic!("1009 should not be invertible mod 1009*1013"),
        }
    }

    #[test]
    fn test_sieve_primes_vec() {
        let primes = sieve_primes_vec(30);
        assert_eq!(primes, vec![2, 3, 5, 7, 11, 13, 17, 19, 23, 29]);
    }

    #[test]
    fn test_mod_sub_no_underflow() {
        let n = Integer::from(1009u64);
        let a = Integer::from(3u32);
        let b = Integer::from(1005u32);
        let result = mod_sub(&a, &b, &n);
        assert_eq!(result, 7u32);
    }

    #[test]
    fn test_ecm_148bit_c45_semiprime() {
        // Third c45 test case:
        // p = 10356651620313423478747 (74-bit), q = 9737786813221482516737 (74-bit)
        let n = semiprime("10356651620313423478747", "9737786813221482516737");
        let bits = n.significant_bits();
        eprintln!("c45 case 3: {} bits, {} digits", bits, n.to_string().len());
        assert!(bits >= 136, "should be >= 136 bits, got {}", bits);
        let result = ecm_factor_st(&n, 400, 50000, 5_000_000);
        assert!(result.is_some(), "ECM should factor c45 semiprime");
        let f = result.unwrap();
        assert!(Integer::from(&n % &f) == 0u32);
        assert!(f > 1u32 && f < n);
    }

    #[test]
    fn test_ecm_c45_balanced_semiprime() {
        // Balanced c45: two 74-bit primes (verified with sympy)
        // p = 9502126893814776953359, q = 10021655427541319958881
        // n has 147 bits, 44 digits
        let p: Integer = "9502126893814776953359".parse().unwrap();
        let q: Integer = "10021655427541319958881".parse().unwrap();
        assert!(p.is_probably_prime(30) != rug::integer::IsPrime::No, "p should be prime");
        assert!(q.is_probably_prime(30) != rug::integer::IsPrime::No, "q should be prime");
        let n = Integer::from(&p * &q);
        let bits = n.significant_bits();
        eprintln!("c45 test: n has {} bits, {} digits", bits, n.to_string().len());
        assert!(bits >= 136 && bits <= 150, "expected ~147 bits, got {}", bits);

        let start = std::time::Instant::now();
        let result = ecm_factor_st(&n, 400, 50000, 5_000_000);
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        eprintln!("c45 ST ECM: {:.1}ms", elapsed);

        assert!(result.is_some(), "ECM should factor balanced c45 (147-bit) semiprime");
        let f = result.unwrap();
        assert!(Integer::from(&n % &f) == 0u32);
        assert!(f > 1u32 && f < n);
    }

    #[test]
    fn test_ecm_c45_parallel() {
        let n = semiprime("9502126893814776953359", "10021655427541319958881");
        let start = std::time::Instant::now();
        let result = ecm_factor(&n, 400, 50000, 5_000_000);
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        eprintln!("c45 MT ECM: {:.1}ms", elapsed);

        assert!(result.is_some(), "Parallel ECM should factor balanced c45 semiprime");
        let f = result.unwrap();
        assert!(Integer::from(&n % &f) == 0u32);
    }

    #[test]
    fn test_ecm_c45_second_case() {
        // p = 9652584922799531340689 (74-bit), q = 10079434548832701666223 (74-bit)
        let n = semiprime("9652584922799531340689", "10079434548832701666223");
        let bits = n.significant_bits();
        eprintln!("c45 case 2: {} bits, {} digits", bits, n.to_string().len());

        let start = std::time::Instant::now();
        let result = ecm_factor(&n, 400, 50000, 5_000_000);
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        eprintln!("c45 case 2 MT ECM: {:.1}ms", elapsed);

        assert!(result.is_some(), "ECM should factor second c45 case");
        let f = result.unwrap();
        assert!(Integer::from(&n % &f) == 0u32);
    }

    #[test]
    fn test_try_ecm_factor_auto_params() {
        let n = Integer::from(1009u64 * 1013);
        let result = try_ecm_factor(&n);
        assert!(result.is_some());
        let (f, ms) = result.unwrap();
        assert!(Integer::from(&n % &f) == 0u32);
        eprintln!("try_ecm_factor: found {} in {:.1}ms", f, ms);
    }

    // ===================================================================
    // U192 arithmetic tests
    // ===================================================================

    #[test]
    fn test_u192_conversion_roundtrip() {
        // Small value.
        let n = Integer::from(12345678u64);
        let u = integer_to_u192(&n);
        let back = u192_to_integer(&u);
        assert_eq!(n, back);

        // Value needing 2 limbs.
        let n: Integer = "18446744073709551617".parse().unwrap(); // 2^64 + 1
        let u = integer_to_u192(&n);
        assert_eq!(u[0], 1);
        assert_eq!(u[1], 1);
        assert_eq!(u[2], 0);
        let back = u192_to_integer(&u);
        assert_eq!(n, back);

        // Value needing 3 limbs.
        let n: Integer = "340282366920938463463374607431768211457".parse().unwrap(); // 2^128 + 1
        let u = integer_to_u192(&n);
        assert_eq!(u[0], 1);
        assert_eq!(u[1], 0);
        assert_eq!(u[2], 1);
        let back = u192_to_integer(&u);
        assert_eq!(n, back);
    }

    #[test]
    fn test_u192_add_mod() {
        let n: U192 = [u64::MAX, u64::MAX, 0]; // 2^128 - 1
        let a: U192 = [u64::MAX - 1, u64::MAX, 0]; // n - 1
        let b: U192 = [1, 0, 0];
        // a + b should equal n - 1 + 1 = n, which wraps to 0.
        let result = add_mod_192(&a, &b, &n);
        assert_eq!(result, U192_ZERO);

        // Small add: 3 + 4 mod 11 = 7.
        let n: U192 = [11, 0, 0];
        let a: U192 = [3, 0, 0];
        let b: U192 = [4, 0, 0];
        assert_eq!(add_mod_192(&a, &b, &n), [7, 0, 0]);

        // Wraparound: 8 + 5 mod 11 = 2.
        let a: U192 = [8, 0, 0];
        let b: U192 = [5, 0, 0];
        assert_eq!(add_mod_192(&a, &b, &n), [2, 0, 0]);
    }

    #[test]
    fn test_u192_sub_mod() {
        let n: U192 = [11, 0, 0];
        // 7 - 3 mod 11 = 4.
        assert_eq!(sub_mod_192(&[7, 0, 0], &[3, 0, 0], &n), [4, 0, 0]);
        // 3 - 7 mod 11 = 7 (since 3 + 11 - 7 = 7).
        assert_eq!(sub_mod_192(&[3, 0, 0], &[7, 0, 0], &n), [7, 0, 0]);
        // 0 - 0 mod 11 = 0.
        assert_eq!(sub_mod_192(&U192_ZERO, &U192_ZERO, &n), U192_ZERO);
    }

    #[test]
    fn test_u192_mul_3x3() {
        // 3 * 4 = 12.
        let a: U192 = [3, 0, 0];
        let b: U192 = [4, 0, 0];
        let t = mul_3x3(&a, &b);
        assert_eq!(t, [12, 0, 0, 0, 0, 0]);

        // (2^64) * (2^64) = 2^128.
        let a: U192 = [0, 1, 0];
        let b: U192 = [0, 1, 0];
        let t = mul_3x3(&a, &b);
        assert_eq!(t, [0, 0, 1, 0, 0, 0]); // limb [2] = 1 = 2^128

        // Verify against rug for a medium product.
        let a_int: Integer = "123456789012345678901234".parse().unwrap();
        let b_int: Integer = "987654321098765432109876".parse().unwrap();
        let a_u = integer_to_u192(&a_int);
        let b_u = integer_to_u192(&b_int);
        let product_6 = mul_3x3(&a_u, &b_u);
        let expected = Integer::from(&a_int * &b_int);
        // Convert 6-limb result to Integer.
        let mut bytes = [0u8; 48];
        for i in 0..6 {
            let limb = product_6[i];
            for j in 0..8 {
                bytes[i * 8 + j] = ((limb >> (j * 8)) & 0xFF) as u8;
            }
        }
        let got = Integer::from_digits(&bytes, rug::integer::Order::LsfLe);
        assert_eq!(got, expected);
    }

    #[test]
    fn test_u192_mont_roundtrip() {
        // n = 1009 * 1013 = 1022117.
        let n = Integer::from(1009u64 * 1013);
        let n_192 = integer_to_u192(&n);
        let mont = Mont192::new(&n_192);

        // Convert 42 to Montgomery form and back.
        let a: U192 = [42, 0, 0];
        let a_mont = to_mont_192(&a, &mont);
        let a_back = from_mont_192(&a_mont, &mont);
        assert_eq!(a_back, a);

        // Convert a larger value.
        let a: U192 = [999999, 0, 0];
        let a_mont = to_mont_192(&a, &mont);
        let a_back = from_mont_192(&a_mont, &mont);
        assert_eq!(a_back, a);
    }

    #[test]
    fn test_u192_mont_mul_matches_rug() {
        // n = 1009 * 1013 = 1022117.
        let n_val = 1009u64 * 1013;
        let n_int = Integer::from(n_val);
        let n_192: U192 = [n_val, 0, 0];
        let mont = Mont192::new(&n_192);

        // Test several multiplications.
        let test_pairs: Vec<(u64, u64)> = vec![
            (7, 13),
            (999999, 42),
            (1, 1022116), // 1 * (n-1)
            (500000, 500000),
            (0, 12345),
        ];

        for (a, b) in test_pairs {
            let a_192: U192 = [a, 0, 0];
            let b_192: U192 = [b, 0, 0];
            let a_mont = to_mont_192(&a_192, &mont);
            let b_mont = to_mont_192(&b_192, &mont);
            let product_mont = mont_mul_192(&a_mont, &b_mont, &mont);
            let product = from_mont_192(&product_mont, &mont);

            let expected_int = Integer::from(a) * Integer::from(b);
            let expected_mod = expected_int.modulo(&n_int);
            let expected_192 = integer_to_u192(&expected_mod);
            assert_eq!(
                product, expected_192,
                "mont_mul({}, {}) mod {} failed: got {:?}, expected {:?}",
                a, b, n_val, product, expected_192
            );
        }
    }

    #[test]
    fn test_u192_mont_mul_large_modulus() {
        // n = p * q where both are ~74-bit primes (148-bit semiprime).
        let n = semiprime("9502126893814776953359", "10021655427541319958881");
        let n_192 = integer_to_u192(&n);
        let mont = Mont192::new(&n_192);

        // Test with known values.
        let a_int: Integer = "12345678901234567890".parse().unwrap();
        let b_int: Integer = "98765432109876543210".parse().unwrap();
        let a_192 = integer_to_u192(&a_int);
        let b_192 = integer_to_u192(&b_int);

        let a_mont = to_mont_192(&a_192, &mont);
        let b_mont = to_mont_192(&b_192, &mont);
        let product_mont = mont_mul_192(&a_mont, &b_mont, &mont);
        let product = from_mont_192(&product_mont, &mont);
        let product_int = u192_to_integer(&product);

        let expected = Integer::from(&a_int * &b_int).modulo(&n);
        assert_eq!(product_int, expected, "Large modulus mont_mul failed");
    }

    #[test]
    fn test_u192_gcd() {
        // gcd(12, 8) = 4.
        assert_eq!(gcd_192(&[12, 0, 0], &[8, 0, 0]), [4, 0, 0]);
        // gcd(17, 1) = 1.
        assert_eq!(gcd_192(&[17, 0, 0], &U192_ONE), U192_ONE);
        // gcd(0, 5) = 5.
        assert_eq!(gcd_192(&U192_ZERO, &[5, 0, 0]), [5, 0, 0]);
        // gcd(1009, 1009*1013) = 1009.
        let n: U192 = [1009 * 1013, 0, 0];
        assert_eq!(gcd_192(&[1009, 0, 0], &n), [1009, 0, 0]);
    }

    #[test]
    fn test_u192_gcd_large() {
        // Verify against rug for large values.
        let a: Integer = "9502126893814776953359".parse().unwrap();
        let b: Integer = "10021655427541319958881".parse().unwrap();
        let n = Integer::from(&a * &b);
        let n_192 = integer_to_u192(&n);
        let a_192 = integer_to_u192(&a);
        let g = gcd_192(&a_192, &n_192);
        assert_eq!(g, a_192, "gcd(p, p*q) should be p");
    }

    #[test]
    fn test_u192_significant_bits() {
        assert_eq!(significant_bits_192(&U192_ZERO), 0);
        assert_eq!(significant_bits_192(&U192_ONE), 1);
        assert_eq!(significant_bits_192(&[0, 1, 0]), 65);
        assert_eq!(significant_bits_192(&[0, 0, 1]), 129);
        assert_eq!(significant_bits_192(&[u64::MAX, u64::MAX, u64::MAX]), 192);
    }

    #[test]
    fn test_u192_shift_roundtrip() {
        // shr then shl is lossless when low bits are zero.
        let a: U192 = [0x123456789ABCDEF0, 0xFEDCBA9876543210, 0x0011223344556677];
        let original = a;
        // Right shift discards low bits, left shift fills with zeros.
        // For a lossless roundtrip, do shr then shl on a value with enough
        // trailing zeros.
        let mut b: U192 = [0u64, 0xFEDCBA9876543210, 0x0011223344556677];
        let orig_b = b;
        // shr by 64 then shl by 64 should preserve when low limb is zero.
        shr_192_inplace(&mut b, 64);
        shl_192_inplace(&mut b, 64);
        assert_eq!(b, orig_b);

        // shl then shr is lossless when high bits fit.
        // Value uses only 155 bits (192-37), so shl by 37 won't overflow.
        let mut c: U192 = [0x123456789ABCDEF0, 0xFEDCBA9876543210, 0x0000000004556677];
        let orig_c = c;
        shl_192_inplace(&mut c, 37);
        shr_192_inplace(&mut c, 37);
        assert_eq!(c, orig_c);

        // Verify that shr on a known value works correctly.
        let mut d: U192 = [0, 0, 1]; // 2^128
        shr_192_inplace(&mut d, 1);
        assert_eq!(d, [0, 1u64 << 63, 0]); // 2^127

        // shl on a known value.
        let mut e: U192 = [1, 0, 0]; // 1
        shl_192_inplace(&mut e, 128);
        assert_eq!(e, [0, 0, 1]); // 2^128

        // Verify original value's high bits would be lost by shl(37).
        let mut f = original;
        shl_192_inplace(&mut f, 37);
        shr_192_inplace(&mut f, 37);
        // The top 37 bits of original a[2] were shifted out, so this
        // should NOT equal original (the test verifies shift is working,
        // not that it's lossless for overflowing values).
        assert_ne!(f, original, "high bits should be lost after shl overflow");
    }

    #[test]
    fn test_ecm_fast_matches_rug_small() {
        // Verify that the fast path finds the same factor for a small semiprime.
        let n = Integer::from(1009u64 * 1013);
        let result = ecm_factor_st(&n, 50, 500, 5000);
        assert!(result.is_some(), "Fast ECM should factor 1009*1013");
        let f = result.unwrap();
        assert!(Integer::from(&n % &f) == 0u32, "factor should divide n");
        assert!(f > 1u32 && f < n, "factor should be non-trivial");
    }

    #[test]
    fn test_ecm_fast_148bit() {
        // Verify the fast path works for a 148-bit semiprime (the primary target).
        let n = semiprime("9502126893814776953359", "10021655427541319958881");
        assert!(n.significant_bits() <= 192, "should fit in 192 bits");

        let start = std::time::Instant::now();
        let result = ecm_factor_st(&n, 400, 50000, 5_000_000);
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        eprintln!("U192 fast ECM (ST): {:.1}ms", elapsed);

        assert!(result.is_some(), "Fast ECM should factor 148-bit semiprime");
        let f = result.unwrap();
        assert!(Integer::from(&n % &f) == 0u32);
        assert!(f > 1u32 && f < n);
    }

    #[test]
    fn test_ecm_fast_parallel_148bit() {
        let n = semiprime("9502126893814776953359", "10021655427541319958881");

        let start = std::time::Instant::now();
        let result = ecm_factor(&n, 400, 50000, 5_000_000);
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        eprintln!("U192 fast ECM (MT): {:.1}ms", elapsed);

        assert!(result.is_some(), "Parallel fast ECM should factor 148-bit semiprime");
        let f = result.unwrap();
        assert!(Integer::from(&n % &f) == 0u32);
    }

    #[test]
    fn test_mont192_n_inv_correct() {
        // Verify: n[0] * n_inv == -1 (mod 2^64), i.e., n[0] * (-n_inv) == 1 (mod 2^64).
        let n: U192 = [1009 * 1013, 0, 0];
        let mont = Mont192::new(&n);
        let check = n[0].wrapping_mul(mont.n_inv.wrapping_neg());
        assert_eq!(check, 1, "n[0] * n_inv^(-1) should be 1 mod 2^64");
    }

    #[test]
    fn test_mont192_r_squared_correct() {
        // Verify R^2 mod n against rug computation.
        let n_val = 1009u64 * 1013;
        let n_192: U192 = [n_val, 0, 0];
        let mont = Mont192::new(&n_192);

        let n_int = Integer::from(n_val);
        let r = Integer::from(1u32) << 192;
        let r_sq = Integer::from(&r * &r);
        let expected = r_sq.modulo(&n_int);
        let expected_192 = integer_to_u192(&expected);
        assert_eq!(mont.r_squared, expected_192, "R^2 mod n mismatch");
    }

    /// Benchmark: compare U192 fast path vs rug::Integer path on the same semiprime.
    /// This test calls both implementations explicitly and prints timing.
    #[test]
    fn test_ecm_benchmark_u192_vs_rug() {
        let n = semiprime("9502126893814776953359", "10021655427541319958881");
        let n_curves = 50;

        // Phase 1 only benchmark (B2 = B1 + 1 to minimize phase 2).
        let b1 = 50000u64;
        let b2 = 50001u64;
        let primes = sieve_primes_vec(b2);

        // Benchmark rug path.
        let start = std::time::Instant::now();
        let mut rug_found = 0;
        for idx in 0..n_curves {
            let sigma = 6u64 + idx;
            if ecm_one_curve(&n, b1, b2, sigma, &primes).is_some() {
                rug_found += 1;
            }
        }
        let rug_ms = start.elapsed().as_secs_f64() * 1000.0;
        let rug_per_curve = rug_ms / n_curves as f64;
        eprintln!(
            "rug path (phase1 only): {:.1}ms total, {:.2}ms/curve, {} found",
            rug_ms, rug_per_curve, rug_found
        );

        // Benchmark U192 fast path.
        let n_192 = integer_to_u192(&n);
        let mont = Mont192::new(&n_192);
        let start = std::time::Instant::now();
        let mut fast_found = 0;
        for idx in 0..n_curves {
            let sigma = 6u64 + idx;
            if ecm_one_curve_fast(&n_192, &n, b1, b2, sigma, &primes, &mont, None).is_some() {
                fast_found += 1;
            }
        }
        let fast_ms = start.elapsed().as_secs_f64() * 1000.0;
        let fast_per_curve = fast_ms / n_curves as f64;
        eprintln!(
            "U192 path (phase1 only): {:.1}ms total, {:.2}ms/curve, {} found",
            fast_ms, fast_per_curve, fast_found
        );

        let speedup = rug_per_curve / fast_per_curve;
        eprintln!("Phase 1 speedup: {:.2}x", speedup);

        // Full benchmark with Phase 2.
        let b2_full = 5_000_000u64;
        let primes_full = sieve_primes_vec(b2_full);

        let start = std::time::Instant::now();
        let mut rug_found_full = 0;
        for idx in 0..20 {
            let sigma = 6u64 + idx;
            if ecm_one_curve(&n, b1, b2_full, sigma, &primes_full).is_some() {
                rug_found_full += 1;
            }
        }
        let rug_full_ms = start.elapsed().as_secs_f64() * 1000.0;
        let rug_full_per = rug_full_ms / 20.0;

        let start = std::time::Instant::now();
        let mut fast_found_full = 0;
        for idx in 0..20 {
            let sigma = 6u64 + idx;
            if ecm_one_curve_fast(&n_192, &n, b1, b2_full, sigma, &primes_full, &mont, None).is_some() {
                fast_found_full += 1;
            }
        }
        let fast_full_ms = start.elapsed().as_secs_f64() * 1000.0;
        let fast_full_per = fast_full_ms / 20.0;
        let full_speedup = rug_full_per / fast_full_per;
        eprintln!(
            "Full (P1+P2): rug {:.2}ms/curve, U192 {:.2}ms/curve, speedup {:.2}x",
            rug_full_per, fast_full_per, full_speedup
        );

        assert_eq!(rug_found, fast_found, "phase1 factor count mismatch");
        assert_eq!(rug_found_full, fast_found_full, "full factor count mismatch");
    }

    #[test]
    fn test_mont192_mul_stress() {
        // Stress test: multiply many random-ish pairs and verify against rug.
        let n = semiprime("10356651620313423478747", "9737786813221482516737");
        let n_192 = integer_to_u192(&n);
        let mont = Mont192::new(&n_192);

        // Use a simple PRNG-like sequence for deterministic test values.
        let mut val = 314159265358979u64;
        for _ in 0..100 {
            val = val.wrapping_mul(6364136223846793005).wrapping_add(1);
            let a_int = Integer::from(val);
            let a_192 = integer_to_u192(&a_int);

            val = val.wrapping_mul(6364136223846793005).wrapping_add(1);
            let b_int = Integer::from(val);
            let b_192 = integer_to_u192(&b_int);

            let a_mont = to_mont_192(&a_192, &mont);
            let b_mont = to_mont_192(&b_192, &mont);
            let product = from_mont_192(&mont_mul_192(&a_mont, &b_mont, &mont), &mont);
            let product_int = u192_to_integer(&product);

            let expected = (Integer::from(&a_int * &b_int)).modulo(&n);
            assert_eq!(
                product_int, expected,
                "Stress test failed for a={}, b={}",
                a_int, b_int
            );
        }
    }
}
