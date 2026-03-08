#![cfg(target_os = "macos")]

use metal::{
    CommandQueue, CompileOptions, ComputePipelineState, Device, Library, MTLResourceOptions,
    MTLSize,
};
use num_bigint::BigUint;
use std::mem;
use std::sync::Arc;

/// A 256-bit integer represented as 8 little-endian 32-bit words,
/// matching the `uint256_t` struct in Metal.
#[repr(C, align(32))]
#[derive(Debug, Clone, Copy, Default)]
pub struct Uint256 {
    pub limbs: [u32; 8],
}

impl Uint256 {
    pub fn from_biguint(val: &BigUint) -> Self {
        let mut limbs = [0u32; 8];
        let bytes = val.to_bytes_le();
        let mut i = 0;
        let mut j = 0;
        while i < bytes.len() && j < 8 {
            let mut word = 0u32;
            for k in 0..4 {
                if i + k < bytes.len() {
                    word |= (bytes[i + k] as u32) << (k * 8);
                }
            }
            limbs[j] = word;
            i += 4;
            j += 1;
        }
        Self { limbs }
    }

    pub fn to_biguint(&self) -> BigUint {
        let mut bytes = Vec::with_capacity(32);
        for &limb in &self.limbs {
            bytes.extend_from_slice(&limb.to_le_bytes());
        }
        BigUint::from_bytes_le(&bytes)
    }
}

/// Matches the `EcmCandidate` struct in Metal.
#[repr(C, align(32))]
#[derive(Debug, Clone, Default)]
pub struct EcmCandidateGpu {
    pub n: Uint256,
    pub x: Uint256,
    pub z: Uint256,
    pub a24: Uint256,
    pub b1: u32,
    pub padding: [u32; 7], // padding to maintain 32-byte alignment for the overall struct length if needed
}

/// Manages the Metal device and pipeline state.
pub struct GpuEcm {
    device: Device,
    command_queue: CommandQueue,
    pipeline_state: ComputePipelineState,
}

impl GpuEcm {
    /// Initializes a new GPU ECM engine.
    /// Returns None if no Metal device is available or compilation fails.
    pub fn new() -> Option<Self> {
        let device = Device::system_default()?;
        let command_queue = device.new_command_queue();

        let source = include_str!("ecm.metal");
        let compile_options = CompileOptions::new();
        let library = device
            .new_library_with_source(source, &compile_options)
            .map_err(|e| {
                println!("Metal compile error: {}", e);
                e
            })
            .ok()?;

        let kernel = library.get_function("batch_ecm", None).unwrap();
        let pipeline_state = device
            .new_compute_pipeline_state_with_function(&kernel)
            .unwrap();

        Some(Self {
            device,
            command_queue,
            pipeline_state,
        })
    }

    /// Dispatches a batch of ECM candidates to the GPU.
    pub fn batch_ecm(&self, candidates: &[EcmCandidateGpu]) -> Vec<Uint256> {
        let count = candidates.len();
        if count == 0 {
            return vec![];
        }

        let cand_bytes = (count * mem::size_of::<EcmCandidateGpu>()) as u64;
        let cand_buffer = self.device.new_buffer_with_data(
            candidates.as_ptr() as *const _,
            cand_bytes,
            MTLResourceOptions::StorageModeShared,
        );

        let res_bytes = (count * mem::size_of::<Uint256>()) as u64;
        let res_buffer = self.device.new_buffer(
            res_bytes,
            MTLResourceOptions::StorageModeShared,
        );

        let command_buffer = self.command_queue.new_command_buffer();
        let compute_encoder = command_buffer.new_compute_command_encoder();

        compute_encoder.set_compute_pipeline_state(&self.pipeline_state);
        compute_encoder.set_buffer(0, Some(&cand_buffer), 0);
        compute_encoder.set_buffer(1, Some(&res_buffer), 0);

        let grid_size = MTLSize {
            width: count as u64,
            height: 1,
            depth: 1,
        };

        let thread_group_size = MTLSize {
            width: std::cmp::min(self.pipeline_state.max_total_threads_per_threadgroup(), count as u64),
            height: 1,
            depth: 1,
        };

        compute_encoder.dispatch_threads(grid_size, thread_group_size);
        compute_encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        let mut results = vec![Uint256::default(); count];
        unsafe {
            let ptr = res_buffer.contents() as *const Uint256;
            std::ptr::copy_nonoverlapping(ptr, results.as_mut_ptr(), count);
        }

        results
    }
}
