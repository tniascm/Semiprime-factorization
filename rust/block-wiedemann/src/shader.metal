#include <metal_stdlib>
using namespace metal;

// In GF(2), addition is XOR (^), multiplication is AND (&).
// A matrix multiplication y = A * x over GF(2) where x and y are vectors of bit-blocks (e.g. 64 vectors computed simultaneously).
// A is a sparse matrix represented in CSR (Compressed Sparse Row) format.

struct SpmvParams {
    uint num_rows;
};

kernel void spmv_gf2_64(
    device const uint* row_ptrs [[buffer(0)]],
    device const uint* col_indices [[buffer(1)]],
    device const ulong* x [[buffer(2)]],
    device ulong* y [[buffer(3)]],
    constant SpmvParams& params [[buffer(4)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= params.num_rows) {
        return;
    }

    uint start = row_ptrs[id];
    uint end = row_ptrs[id + 1];

    ulong acc = 0;

    // Compute the dot product for row `id`.
    // Since the sparse matrix A has 1s at `col_indices`, we just XOR the elements of x at those indices.
    // This effectively computes A * x for 64 right-hand sides simultaneously (since x and acc are 64-bit).
    for (uint i = start; i < end; ++i) {
        uint col = col_indices[i];
        acc ^= x[col];
    }

    y[id] = acc;
}
