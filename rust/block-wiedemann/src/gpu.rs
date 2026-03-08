//! Metal GPU implementation for GF(2) SpMV.

use crate::matrix::CsrMatrix;
use crate::WiedemannError;
use metal::*;
use std::mem;

#[repr(C)]
struct SpmvParams {
    num_rows: u32,
}

pub struct GpuContext {
    device: Device,
    command_queue: CommandQueue,
    pipeline_state: ComputePipelineState,
}

impl GpuContext {
    pub fn new() -> Result<Option<Self>, WiedemannError> {
        let device = match Device::system_default() {
            Some(dev) => dev,
            None => return Ok(None), // No Metal support (fallback to CPU)
        };

        let command_queue = device.new_command_queue();

        let library_path = std::env::var("METAL_SHADER_PATH")
            .unwrap_or_else(|_| "src/shader.metal".to_string());

        let library = match device.new_library_with_source(
            include_str!("shader.metal"),
            &CompileOptions::new(),
        ) {
            Ok(lib) => lib,
            Err(e) => return Err(WiedemannError::MetalError(format!("Compile failed: {}", e))),
        };

        let function = library.get_function("spmv_gf2_64", None).unwrap();
        let pipeline_state = match device.new_compute_pipeline_state_with_function(&function) {
            Ok(state) => state,
            Err(e) => return Err(WiedemannError::MetalError(format!("Pipeline failed: {}", e))),
        };

        Ok(Some(Self {
            device,
            command_queue,
            pipeline_state,
        }))
    }

    /// Run a sequence generation for Block Wiedemann.
    ///
    /// Computes v_k = (A^T * A)^k * v_0 for k = 1..iterations.
    /// In GF(2), vector components are bits, so we pack 64 sequence vectors into u64 arrays.
    pub fn compute_sequence(
        &self,
        matrix_a: &CsrMatrix,
        matrix_at: &CsrMatrix,
        initial_vector: &[u64],
        iterations: usize,
    ) -> Result<Vec<Vec<u64>>, WiedemannError> {
        let n = matrix_a.cols;
        let m = matrix_a.rows;

        if initial_vector.len() != n {
            return Err(WiedemannError::DimensionMismatch("Vector length must match cols".into()));
        }

        // Create buffers for A
        let a_row_ptrs_buf = self.device.new_buffer_with_data(
            matrix_a.row_ptrs.as_ptr() as *const _,
            (matrix_a.row_ptrs.len() * mem::size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let a_col_idx_buf = self.device.new_buffer_with_data(
            matrix_a.col_indices.as_ptr() as *const _,
            (matrix_a.col_indices.len() * mem::size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Create buffers for A^T
        let at_row_ptrs_buf = self.device.new_buffer_with_data(
            matrix_at.row_ptrs.as_ptr() as *const _,
            (matrix_at.row_ptrs.len() * mem::size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let at_col_idx_buf = self.device.new_buffer_with_data(
            matrix_at.col_indices.as_ptr() as *const _,
            (matrix_at.col_indices.len() * mem::size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Vector buffers (ping-pong)
        let vec_size = (n * mem::size_of::<u64>()) as u64;
        let intermediate_size = (m * mem::size_of::<u64>()) as u64;

        let mut v_in = self.device.new_buffer_with_data(
            initial_vector.as_ptr() as *const _,
            vec_size,
            MTLResourceOptions::StorageModeShared,
        );
        let v_mid = self.device.new_buffer(intermediate_size, MTLResourceOptions::StorageModeShared);
        let mut v_out = self.device.new_buffer(vec_size, MTLResourceOptions::StorageModeShared);

        let mut results = Vec::with_capacity(iterations);

        let thread_group_count = MTLSize { width: 256, height: 1, depth: 1 };
        let thread_groups_a = MTLSize {
            width: (m as u64 + 255) / 256,
            height: 1,
            depth: 1
        };
        let thread_groups_at = MTLSize {
            width: (n as u64 + 255) / 256,
            height: 1,
            depth: 1
        };

        for _ in 0..iterations {
            let command_buffer = self.command_queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();

            encoder.set_compute_pipeline_state(&self.pipeline_state);

            // Step 1: v_mid = A * v_in
            let params_a = SpmvParams { num_rows: m as u32 };
            encoder.set_buffer(0, Some(&a_row_ptrs_buf), 0);
            encoder.set_buffer(1, Some(&a_col_idx_buf), 0);
            encoder.set_buffer(2, Some(&v_in), 0);
            encoder.set_buffer(3, Some(&v_mid), 0);
            encoder.set_bytes(4, mem::size_of::<SpmvParams>() as u64, &params_a as *const _ as *const _);

            encoder.dispatch_thread_groups(thread_groups_a, thread_group_count);

            // Step 2: v_out = A^T * v_mid
            let params_at = SpmvParams { num_rows: n as u32 };
            encoder.set_buffer(0, Some(&at_row_ptrs_buf), 0);
            encoder.set_buffer(1, Some(&at_col_idx_buf), 0);
            encoder.set_buffer(2, Some(&v_mid), 0);
            encoder.set_buffer(3, Some(&v_out), 0);
            encoder.set_bytes(4, mem::size_of::<SpmvParams>() as u64, &params_at as *const _ as *const _);

            encoder.dispatch_thread_groups(thread_groups_at, thread_group_count);

            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();

            // Read back result
            let mut out_vec = vec![0u64; n];
            unsafe {
                std::ptr::copy_nonoverlapping(
                    v_out.contents() as *const u64,
                    out_vec.as_mut_ptr(),
                    n,
                );
            }
            results.push(out_vec);

            // Swap buffers
            std::mem::swap(&mut v_in, &mut v_out);
        }

        Ok(results)
    }
}
