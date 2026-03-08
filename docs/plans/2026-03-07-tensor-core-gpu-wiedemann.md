# Tensor Core GPU Acceleration Plan for Block Wiedemann

**Date:** 2026-03-07
**Component:** `rust-nfs` / `block-wiedemann`

## Overview
This document details the strategy for moving the core $O(n^2)$ Sparse Matrix-Vector (SpMV) operations of the GF(2) Block Wiedemann algorithm from CPU (`par_mul_block_gf2`) to Tensor Core-accelerated GPU kernels (NVIDIA CUDA and Apple Metal).

## The Mathematical Mapping to INT8
Tensor Cores (like those on NVIDIA Ampere/Hopper or Apple M3 Max) do not natively support GF(2) matrix multiplication. They support operations like INT8 $C = A \times B + C$.

To utilize them for GF(2):
1.  **GF(2) Addition is XOR:** Standard addition `+` naturally overflows and carries. However, the lowest bit (LSB) of integer addition is mathematically equivalent to XOR ($a \oplus b = (a + b) \pmod 2$).
2.  **GF(2) Multiplication is AND:** The LSB of integer multiplication is equivalent to AND ($a \land b = (a \cdot b) \pmod 2$).
3.  **The INT8 Trick:** We can pack GF(2) elements into the LSBs of INT8 values. We instruct the Tensor Core to perform a massive INT8 matrix-matrix multiplication block. The resulting accumulation matrix will contain integer sums. We then simply read the lowest bit (`result & 1`) to retrieve the exact GF(2) result.

## CUDA Implementation Strategy (`mma.sync`)

1.  **Matrix Blocking:** The sparse matrix $M$ must be converted from Compressed Sparse Row (CSR) into a Blocked CSR (BCSR) or ELLPACK format, aligned to the Tensor Core warp sizes (e.g., $16 \times 16$ or $32 \times 8$).
2.  **Vector Blocking:** The 64 vectors (currently packed into `u64` words) are unpacked into INT8 arrays in GPU shared memory.
3.  **Kernel Execution:**
    *   Load the sparse matrix block into registers.
    *   Load the vector block into registers.
    *   Execute `mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32` (or similar PTX instruction).
    *   Extract the LSB of the resulting INT32 accumulators and write back to global memory.

## Apple Metal Implementation Strategy (`simdgroup_matrix`)

Apple Silicon provides similar matrix-multiplication coprocessors accessible via the Metal Shading Language (MSL) `simdgroup_matrix` types.

1.  **Threadgroup Memory:** Load segments of the sparse matrix into `threadgroup` memory to hide memory latency.
2.  **SIMD Matrix Math:** Use `simdgroup_matrix<char, 8, 8>` (INT8 matrices) to perform the multiplication.
3.  **The Kernel:**
    ```cpp
    simdgroup_matrix<char, 8, 8> matrix_A;
    simdgroup_matrix<char, 8, 8> matrix_B;
    simdgroup_matrix<int, 8, 8> result_C;

    // Load from memory
    simdgroup_load(matrix_A, ...);
    simdgroup_load(matrix_B, ...);

    // Tensor Core Multiplication
    simdgroup_multiply_accumulate(result_C, matrix_A, matrix_B, result_C);

    // Extract GF(2) parity
    // ...
    ```

## Rust FFI Integration

The `block-wiedemann` crate will expose a generic interface for the SpMV step.
We will introduce `#[cfg(feature = "cuda")]` and `#[cfg(feature = "metal")]` flags.

When a GPU feature is enabled:
1.  The `SparseMatrixGF2` (which implements `serde::Serialize`) will be serialized and pushed directly to the GPU memory space once.
2.  The `par_mul_block_gf2` function will be bypassed. Instead, an FFI call will invoke the CUDA/Metal kernel, passing pointers to the input and output vector blocks.
3.  The zero-copy architecture ensures no CPU-GPU memory bottleneck occurs during the critical sequence generation loop.
