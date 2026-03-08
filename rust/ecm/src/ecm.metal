#include <metal_stdlib>
using namespace metal;

// 256-bit integer represented as 8x 32-bit limbs (little-endian)
struct uint256_t {
    uint limbs[8];
};

struct alignas(32) EcmCandidate {
    uint256_t n;
    uint256_t x;
    uint256_t z;
    uint256_t a24;
    uint b1;
    uint padding[7];
};

bool add_carry(thread uint256_t& res, thread const uint256_t& a, thread const uint256_t& b) {
    uint carry = 0;
    for (int i = 0; i < 8; i++) {
        uint sum1 = a.limbs[i] + b.limbs[i];
        uint carry1 = sum1 < a.limbs[i] ? 1 : 0;
        uint sum2 = sum1 + carry;
        uint carry2 = sum2 < sum1 ? 1 : 0;
        res.limbs[i] = sum2;
        carry = carry1 | carry2;
    }
    return carry != 0;
}

bool sub_borrow(thread uint256_t& res, thread const uint256_t& a, thread const uint256_t& b) {
    uint borrow = 0;
    for (int i = 0; i < 8; i++) {
        uint diff1 = a.limbs[i] - b.limbs[i];
        uint borrow1 = a.limbs[i] < b.limbs[i] ? 1 : 0;
        uint diff2 = diff1 - borrow;
        uint borrow2 = diff1 < borrow ? 1 : 0;
        res.limbs[i] = diff2;
        borrow = borrow1 | borrow2;
    }
    return borrow != 0;
}

int cmp(thread const uint256_t& a, thread const uint256_t& b) {
    for (int i = 7; i >= 0; i--) {
        if (a.limbs[i] > b.limbs[i]) return 1;
        if (a.limbs[i] < b.limbs[i]) return -1;
    }
    return 0;
}

void add_mod(thread uint256_t& res, thread const uint256_t& a, thread const uint256_t& b, thread const uint256_t& m) {
    uint256_t temp;
    bool carry = add_carry(temp, a, b);
    if (carry || cmp(temp, m) >= 0) {
        sub_borrow(res, temp, m);
    } else {
        res = temp;
    }
}

void sub_mod(thread uint256_t& res, thread const uint256_t& a, thread const uint256_t& b, thread const uint256_t& m) {
    uint256_t temp;
    bool borrow = sub_borrow(temp, a, b);
    if (borrow) {
        add_carry(res, temp, m);
    } else {
        res = temp;
    }
}

// Interleaved multiplication and reduction (Montgomery-like or simple shift-and-add)
void mul_mod(thread uint256_t& res, thread const uint256_t& a, thread const uint256_t& b, thread const uint256_t& m) {
    uint256_t t = {0};

    for (int i = 255; i >= 0; i--) {
        // t = (t << 1) % m
        bool carry = add_carry(t, t, t);
        if (carry || cmp(t, m) >= 0) {
            sub_borrow(t, t, m);
        }

        // if bit i of b is 1, t = (t + a) % m
        uint word = b.limbs[i / 32];
        uint bit = (word >> (i % 32)) & 1;
        if (bit) {
            add_mod(t, t, a, m);
        }
    }
    res = t;
}

// Montgomery ladder step on Montgomery curve: By^2 = x^3 + Ax^2 + x
// Projective coordinates (X : Z)
void montgomery_step(thread uint256_t& x1, thread uint256_t& z1, thread uint256_t& x2, thread uint256_t& z2, thread const uint256_t& x_diff, thread const uint256_t& z_diff, thread const uint256_t& a24, thread const uint256_t& n, uint bit) {
    uint256_t a, b, c, d, e, aa, bb, da, cb;

    add_mod(a, x1, z1, n);
    sub_mod(b, x1, z1, n);
    add_mod(c, x2, z2, n);
    sub_mod(d, x2, z2, n);

    mul_mod(da, d, a, n);
    mul_mod(cb, c, b, n);

    uint256_t da_plus_cb, da_minus_cb;
    add_mod(da_plus_cb, da, cb, n);
    sub_mod(da_minus_cb, da, cb, n);

    mul_mod(aa, a, a, n);
    mul_mod(bb, b, b, n);

    sub_mod(e, aa, bb, n);

    if (bit == 0) {
        mul_mod(x1, aa, bb, n);

        uint256_t a24_e;
        mul_mod(a24_e, a24, e, n);
        uint256_t bb_plus_a24_e;
        add_mod(bb_plus_a24_e, bb, a24_e, n);
        mul_mod(z1, e, bb_plus_a24_e, n);

        mul_mod(x2, da_plus_cb, da_plus_cb, n);

        uint256_t t;
        mul_mod(t, da_minus_cb, da_minus_cb, n);
        mul_mod(z2, x_diff, t, n); // assuming z_diff = 1 for simplification in basic ladder
    } else {
        mul_mod(x2, aa, bb, n);

        uint256_t a24_e;
        mul_mod(a24_e, a24, e, n);
        uint256_t bb_plus_a24_e;
        add_mod(bb_plus_a24_e, bb, a24_e, n);
        mul_mod(z2, e, bb_plus_a24_e, n);

        mul_mod(x1, da_plus_cb, da_plus_cb, n);

        uint256_t t;
        mul_mod(t, da_minus_cb, da_minus_cb, n);
        mul_mod(z1, x_diff, t, n);
    }
}

kernel void batch_ecm(
    device const EcmCandidate* candidates [[buffer(0)]],
    device uint256_t* results [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    EcmCandidate c = candidates[id];

    // Start with P = (x, z)
    uint256_t P_x = c.x;
    uint256_t P_z = c.z;

    // Iterate over integers up to B1 (simple prime test)
    for (uint p = 2; p <= c.b1; p++) {
        bool is_prime = true;
        for (uint i = 2; i * i <= p; i++) {
            if (p % i == 0) {
                is_prime = false;
                break;
            }
        }

        if (!is_prime) continue;

        // Scalar multiplication: k = p
        uint k = p;
        // In ECM, k is usually p^(floor(log_p(B1)))
        // For simplicity in this kernel, we just multiply by p.
        // A full implementation would compute the maximal power of p <= B1.
        uint max_pow = p;
        while (max_pow * p <= c.b1) {
            max_pow *= p;
        }
        k = max_pow;

        // Montgomery ladder for scalar k
        uint256_t R0_x = {0}; R0_x.limbs[0] = 1; // Point at infinity (Z=0)
        uint256_t R0_z = {0};
        uint256_t R1_x = P_x;
        uint256_t R1_z = P_z;

        // Find highest set bit
        int top_bit = 31;
        while (top_bit >= 0 && ((k >> top_bit) & 1) == 0) {
            top_bit--;
        }

        if (top_bit >= 0) {
            for (int i = top_bit; i >= 0; i--) {
                uint bit = (k >> i) & 1;
                montgomery_step(R0_x, R0_z, R1_x, R1_z, P_x, P_z, c.a24, c.n, bit);
            }
        }

        // Update P for the next prime
        P_x = R0_x;
        P_z = R0_z;
    }

    // Result is placed in output buffer (X-coordinate)
    results[id] = P_x;
}
