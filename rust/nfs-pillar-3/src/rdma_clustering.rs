use std::sync::{Arc, Mutex};
use std::collections::HashMap;
use thiserror::Error;
use log::{info, debug, error};

/// Errors that can occur during RDMA operations.
#[derive(Error, Debug, Clone, PartialEq)]
pub enum RdmaError {
    #[error("Invalid memory region key (rkey mismatch)")]
    InvalidRkey,
    #[error("Write operation out of bounds of the memory region")]
    OutOfBounds,
    #[error("Remote memory region not found")]
    MrNotFound,
    #[error("RDMA Hardware failure: {0}")]
    HardwareFailure(String),
}

/// Simulates a registered RDMA memory region (Memory Region / MR)
/// In a real system, this would be a pinned memory buffer registered with the NIC via ibverbs.
pub struct MemoryRegion {
    pub buffer: Vec<u8>,
    pub rkey: u32,
    pub addr: u64,
}

impl MemoryRegion {
    pub fn new(size: usize, rkey: u32, addr: u64) -> Self {
        MemoryRegion {
            buffer: vec![0; size],
            rkey,
            addr,
        }
    }
}

/// Represents a remote node's memory region metadata that allows RDMA ops.
#[derive(Clone, Copy, Debug)]
pub struct RemoteRegion {
    pub rkey: u32,
    pub addr: u64,
}

/// A trait defining the hardware interface for RDMA networking.
/// This allows swapping the simulated NIC with a real `ibverbs` provider in production.
pub trait RdmaProvider: Send + Sync {
    /// Registers a memory region with the NIC, pinning it for zero-copy access.
    fn register_mr(&self, mr: Arc<Mutex<MemoryRegion>>);

    /// Performs an RDMA write operation to a remote memory region.
    fn rdma_write(&self, remote_mr: RemoteRegion, offset: usize, data: &[u8]) -> Result<(), RdmaError>;
}

/// Simulates a minimal RDMA network interface card (NIC)
/// facilitating direct memory access without remote CPU involvement.
pub struct SimulatedNic {
    memory_regions: Arc<Mutex<HashMap<u64, Arc<Mutex<MemoryRegion>>>>>,
}

impl Default for SimulatedNic {
    fn default() -> Self {
        Self::new()
    }
}

impl SimulatedNic {
    pub fn new() -> Self {
        SimulatedNic {
            memory_regions: Arc::new(Mutex::new(HashMap::new())),
        }
    }
}

impl RdmaProvider for SimulatedNic {
    fn register_mr(&self, mr: Arc<Mutex<MemoryRegion>>) {
        let mut regions = self.memory_regions.lock().unwrap();
        let addr = {
            let mr_lock = mr.lock().unwrap();
            mr_lock.addr
        };
        debug!("Registering MR at address {:#X}", addr);
        regions.insert(addr, mr);
    }

    /// Simulates RDMA Write: writes `data` directly to the `remote_addr`.
    /// Does not notify the remote CPU (zero-copy networking).
    fn rdma_write(&self, remote_mr: RemoteRegion, offset: usize, data: &[u8]) -> Result<(), RdmaError> {
        let regions = self.memory_regions.lock().unwrap();
        if let Some(mr_arc) = regions.get(&remote_mr.addr) {
            let mut mr = mr_arc.lock().unwrap();
            if mr.rkey != remote_mr.rkey {
                error!("RDMA Write failed: Invalid rkey");
                return Err(RdmaError::InvalidRkey);
            }
            if offset + data.len() > mr.buffer.len() {
                error!("RDMA Write failed: Out of bounds");
                return Err(RdmaError::OutOfBounds);
            }

            // Zero-copy network transfer simulated by a local memory copy
            mr.buffer[offset..offset + data.len()].copy_from_slice(data);
            debug!("RDMA Write successful: {} bytes written to {:#X} offset {}", data.len(), remote_mr.addr, offset);
            Ok(())
        } else {
            error!("RDMA Write failed: MR not found at {:#X}", remote_mr.addr);
            Err(RdmaError::MrNotFound)
        }
    }
}

/// A node in the clustering system (could be Master or Worker).
pub struct ClusterNode {
    pub local_mr: Arc<Mutex<MemoryRegion>>,
    pub nic: Arc<dyn RdmaProvider>,
}

impl ClusterNode {
    pub fn new(nic: Arc<dyn RdmaProvider>, mr_size: usize, addr: u64, rkey: u32) -> Self {
        let mr = Arc::new(Mutex::new(MemoryRegion::new(mr_size, rkey, addr)));
        nic.register_mr(Arc::clone(&mr));
        info!("Initialized ClusterNode with local MR of {} bytes", mr_size);
        ClusterNode { local_mr: mr, nic }
    }

    pub fn get_remote_region_info(&self) -> RemoteRegion {
        let mr = self.local_mr.lock().unwrap();
        RemoteRegion { rkey: mr.rkey, addr: mr.addr }
    }

    /// Stream an array of 64-bit relations directly into the master's memory.
    /// In the NFS pipeline, this happens synchronously and lock-free across the PCIe/Infiniband bus.
    pub fn stream_relations_to_master(&self, master_region: RemoteRegion, offset_bytes: usize, relations: &[u64]) -> Result<(), RdmaError> {
        // Serialize to bytes (representing DMA payload)
        let mut data = Vec::with_capacity(relations.len() * 8);
        for rel in relations {
            data.extend_from_slice(&rel.to_ne_bytes());
        }
        self.nic.rdma_write(master_region, offset_bytes, &data)
    }

    /// Read raw relation bytes from the local RDMA registered region.
    pub fn read_local_relations(&self, count: usize) -> Vec<u64> {
        let mr = self.local_mr.lock().unwrap();
        let mut rels = Vec::with_capacity(count);
        for i in 0..count {
            let start = i * 8;
            let end = start + 8;
            if end > mr.buffer.len() {
                break;
            }
            let mut buf = [0u8; 8];
            buf.copy_from_slice(&mr.buffer[start..end]);
            rels.push(u64::from_ne_bytes(buf));
        }
        rels
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_copy_rdma_pipeline() {
        // Initialize mock NIC provider
        let nic: Arc<dyn RdmaProvider> = Arc::new(SimulatedNic::new());

        // Master node allocates 1024 bytes (128 relations)
        let master = ClusterNode::new(Arc::clone(&nic), 1024, 0x1000, 42);
        let master_region = master.get_remote_region_info();

        // Worker node does not need a large MR for receiving
        let worker1 = ClusterNode::new(Arc::clone(&nic), 8, 0x2000, 43);
        let worker2 = ClusterNode::new(Arc::clone(&nic), 8, 0x3000, 44);

        let w1_rels: Vec<u64> = vec![1, 2, 3];
        let w2_rels: Vec<u64> = vec![99, 100];

        // Workers perform RDMA write directly into Master's memory at specific offsets
        assert!(worker1.stream_relations_to_master(master_region, 0, &w1_rels).is_ok());

        // Worker 2 writes after Worker 1's data (offset 24 bytes)
        assert!(worker2.stream_relations_to_master(master_region, 24, &w2_rels).is_ok());

        // Master reads from its local memory without parsing files or blocking on channels
        let read_rels = master.read_local_relations(5);
        assert_eq!(read_rels, vec![1, 2, 3, 99, 100]);
    }

    #[test]
    fn test_rdma_errors() {
        let nic: Arc<dyn RdmaProvider> = Arc::new(SimulatedNic::new());
        let master = ClusterNode::new(Arc::clone(&nic), 16, 0x1000, 42); // Only 16 bytes
        let worker = ClusterNode::new(Arc::clone(&nic), 8, 0x2000, 43);

        let mut invalid_region = master.get_remote_region_info();
        invalid_region.rkey = 999; // Wrong key

        let data = vec![1u64];
        let result = worker.stream_relations_to_master(invalid_region, 0, &data);
        assert_eq!(result, Err(RdmaError::InvalidRkey));

        let valid_region = master.get_remote_region_info();
        let large_data = vec![1, 2, 3]; // 24 bytes, exceeds 16
        let result2 = worker.stream_relations_to_master(valid_region, 0, &large_data);
        assert_eq!(result2, Err(RdmaError::OutOfBounds));

        let mut unknown_region = valid_region;
        unknown_region.addr = 0x9999;
        let result3 = worker.stream_relations_to_master(unknown_region, 0, &data);
        assert_eq!(result3, Err(RdmaError::MrNotFound));
    }
}
